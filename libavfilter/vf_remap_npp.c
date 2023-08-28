/**
 * @file
 * Pixel remap filter
 * This filter copies pixel by pixel a source frame to a target frame.
 * It remaps the pixels to a new x,y destination based on two files ymap/xmap.
 * Map files are passed as a parameter and are in PGM format (P2 or P5),
 * where the values are y(rows)/x(cols) coordinates of the source_frame.
 * The *target* frame dimension is based on mapfile dimensions: specified in the
 * header of the mapfile and reflected in the number of datavalues.
 * Dimensions of ymap and xmap must be equal. Datavalues must be positive or zero.
 * Any datavalue in the ymap or xmap which value is higher
 * then the *source* frame height or width is silently ignored, leaving a
 * blank/chromakey pixel. This can safely be used as a feature to create overlays.
 *
 * Algorithm digest:
 * Target_frame[y][x] = Source_frame[ ymap[y][x] ][ [xmap[y][x] ];
 */

#include <nppi.h>
#include <stdio.h>
#include <string.h>

#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/parseutils.h"
#include "libavutil/eval.h"
#include "libavutil/colorspace.h"
#include "libavutil/pixdesc.h"
#include "libavutil/opt.h"

#include "avfilter.h"
#include "drawutils.h"
#include "framesync.h"
#include "internal.h"
#include "video.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, device_hwctx->internal->cuda_dl, x)

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUV444P,
};

typedef struct RemapNPPContext {
    
    const AVClass* class;

    AVBufferRef* frames_ctx;
    AVFrame* own_frame;
    AVFrame* tmp_frame;

    NppiBorderType border_type;
    int interp_algo;
} RemapNPPContext;

#define OFFSET(x) offsetof(RemapNPPContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption remap_npp_options[] = {
    { "interp_algo", "Interpolation algorithm used for resizing", OFFSET(interp_algo), AV_OPT_TYPE_INT, { .i64 = NPPI_INTER_CUBIC }, 0, INT_MAX, FLAGS, "interp_algo" },
        { "nn",                 "nearest neighbour",                 0, AV_OPT_TYPE_CONST, { .i64 = NPPI_INTER_NN                 }, 0, 0, FLAGS, "interp_algo" },
        { "linear",             "linear",                            0, AV_OPT_TYPE_CONST, { .i64 = NPPI_INTER_LINEAR             }, 0, 0, FLAGS, "interp_algo" },
        { "cubic",              "cubic",                             0, AV_OPT_TYPE_CONST, { .i64 = NPPI_INTER_CUBIC              }, 0, 0, FLAGS, "interp_algo" },
        { "cubic2p_bspline",    "2-parameter cubic (B=1, C=0)",      0, AV_OPT_TYPE_CONST, { .i64 = NPPI_INTER_CUBIC2P_BSPLINE    }, 0, 0, FLAGS, "interp_algo" },
        { "cubic2p_catmullrom", "2-parameter cubic (B=0, C=1/2)",    0, AV_OPT_TYPE_CONST, { .i64 = NPPI_INTER_CUBIC2P_CATMULLROM }, 0, 0, FLAGS, "interp_algo" },
        { "cubic2p_b05c03",     "2-parameter cubic (B=1/2, C=3/10)", 0, AV_OPT_TYPE_CONST, { .i64 = NPPI_INTER_CUBIC2P_B05C03     }, 0, 0, FLAGS, "interp_algo" },
        { "super",              "supersampling",                     0, AV_OPT_TYPE_CONST, { .i64 = NPPI_INTER_SUPER              }, 0, 0, FLAGS, "interp_algo" },
        { "lanczos",            "Lanczos",                           0, AV_OPT_TYPE_CONST, { .i64 = NPPI_INTER_LANCZOS            }, 0, 0, FLAGS, "interp_algo" },

    { "border_type", "Type of operation to be performed on image border", OFFSET(border_type), AV_OPT_TYPE_INT, { .i64 = NPP_BORDER_REPLICATE }, NPP_BORDER_REPLICATE, NPP_BORDER_REPLICATE, FLAGS, "border_type" },
        { "replicate", "replicate pixels", 0, AV_OPT_TYPE_CONST, { .i64 = NPP_BORDER_REPLICATE }, 0, 0, FLAGS, "border_type" },
    { NULL }
    
};

AVFILTER_DEFINE_CLASS(remap_npp);

static int remap_npp_init(AVFilterContext* ctx)
{
    av_log(ctx, AV_LOG_DEBUG, "remap_npp_init\n");

    RemapNPPContext* s = ctx->priv;

    s->own_frame = av_frame_alloc();
    if (!s->own_frame)
        goto fail;

    s->tmp_frame = av_frame_alloc();
    if (!s->tmp_frame)
        goto fail;

    return 0;

fail:
    av_frame_free(&s->own_frame);
    av_frame_free(&s->tmp_frame);
    return AVERROR(ENOMEM);
}

static int remap_npp_config(AVFilterContext* ctx, int width, int height)
{
    av_log(ctx, AV_LOG_DEBUG, "remap_npp_config\n");

    RemapNPPContext* s = ctx->priv;
    AVHWFramesContext *out_ctx, *in_ctx;
    int i, ret, supported_format = 0;

    if (!ctx->inputs[0]->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        goto fail;
    }

    in_ctx = (AVHWFramesContext*)ctx->inputs[0]->hw_frames_ctx->data;

    s->frames_ctx = av_hwframe_ctx_alloc(in_ctx->device_ref);
    if (!s->frames_ctx)
        goto fail;

    out_ctx = (AVHWFramesContext*)s->frames_ctx->data;
    out_ctx->format = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = in_ctx->sw_format;
    out_ctx->width = FFALIGN(width, 32);
    out_ctx->height = FFALIGN(height, 32);

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++) {
        if (in_ctx->sw_format == supported_formats[i]) {
            supported_format = 1;
            break;
        }
    }

    if (!supported_format) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n", av_get_pix_fmt_name(in_ctx->sw_format));
        goto fail;
    }

    ret = av_hwframe_ctx_init(s->frames_ctx);
    if (ret < 0)
        goto fail;

    ret = av_hwframe_get_buffer(s->frames_ctx, s->own_frame, 0);
    if (ret < 0)
        goto fail;

    ctx->outputs[0]->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!ctx->outputs[0]->hw_frames_ctx)
        goto fail;

    return 0;

fail:
    av_buffer_unref(&s->frames_ctx);
    return AVERROR(ENOMEM);
}

static void remap_npp_uninit(AVFilterContext* ctx)
{
    av_log(ctx, AV_LOG_DEBUG, "remap_npp_uninit\n");

    RemapNPPContext* s = ctx->priv;

    av_buffer_unref(&s->frames_ctx);
    av_frame_free(&s->own_frame);
    av_frame_free(&s->tmp_frame);
}

static int remap_npp_config_props(AVFilterLink* outlink)
{
    AVFilterLink* inlink = outlink->src->inputs[0];

    outlink->w = inlink->w;
    outlink->h = inlink->h;

    if (inlink->sample_aspect_ratio.num)
        outlink->sample_aspect_ratio = av_mul_q(
            (AVRational){outlink->h * inlink->w, outlink->w * inlink->h},
            inlink->sample_aspect_ratio);
    else
        outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;

    remap_npp_config(outlink->src, inlink->w, inlink->h);

    return 0;
}

static int nppsharpen_sharpen(AVFilterContext* ctx, AVFrame* out, AVFrame* in)
{
    AVHWFramesContext* in_ctx =
        (AVHWFramesContext*)ctx->inputs[0]->hw_frames_ctx->data;
    RemapNPPContext* s = ctx->priv;

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(in_ctx->sw_format);

    for (int i = 0; i < FF_ARRAY_ELEMS(in->data) && in->data[i]; i++) {
        int ow = AV_CEIL_RSHIFT(in->width, (i == 1 || i == 2) ? desc->log2_chroma_w : 0);
        int oh = AV_CEIL_RSHIFT(in->height, (i == 1 || i == 2) ? desc->log2_chroma_h : 0);

        NppStatus err = nppiFilterSharpenBorder_8u_C1R(
            in->data[i], in->linesize[i], (NppiSize){ow, oh}, (NppiPoint){0, 0},
            out->data[i], out->linesize[i], (NppiSize){ow, oh}, s->border_type);
        if (err != NPP_SUCCESS) {
            av_log(ctx, AV_LOG_ERROR, "NPP sharpen error: %d\n", err);
            return AVERROR_EXTERNAL;
        }
    }

    return 0;
}

static int remap_npp_filter_frame(AVFilterLink* link, AVFrame* in)
{
    AVFilterContext* ctx = link->dst;
    RemapNPPContext* s = ctx->priv;
    AVFilterLink* outlink = ctx->outputs[0];
    AVHWFramesContext* frames_ctx =
        (AVHWFramesContext*)outlink->hw_frames_ctx->data;
    AVCUDADeviceContext* device_hwctx = frames_ctx->device_ctx->hwctx;

    AVFrame* out = NULL;
    CUcontext dummy;
    int ret = 0;

    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    ret = CHECK_CU(device_hwctx->internal->cuda_dl->cuCtxPushCurrent(
        device_hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;

    ret = nppsharpen_sharpen(ctx, s->own_frame, in);
    if (ret < 0)
        goto pop_ctx;

    ret = av_hwframe_get_buffer(s->own_frame->hw_frames_ctx, s->tmp_frame, 0);
    if (ret < 0)
        goto pop_ctx;

    av_frame_move_ref(out, s->own_frame);
    av_frame_move_ref(s->own_frame, s->tmp_frame);

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        goto pop_ctx;

    av_frame_free(&in);

pop_ctx:
    CHECK_CU(device_hwctx->internal->cuda_dl->cuCtxPopCurrent(&dummy));
    if (!ret)
        return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

static const AVFilterPad remap_npp_inputs[] = {
    {
        .name         = "top",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = remap_npp_filter_frame,
    },
#if 0
    {
        .name         = "bottom",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = remap_npp_config_props,
    },
//#if 0
    {
        .name         = "xmap",
        .type         = AVMEDIA_TYPE_VIDEO,
//        .config_props = remap_npp_config_props,
    },
    {
        .name         = "ymap",
        .type         = AVMEDIA_TYPE_VIDEO,
//        .config_props = remap_npp_config_props,
    },
#endif
};

static const AVFilterPad remap_npp_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = remap_npp_config_props,
    },
};

const AVFilter ff_vf_remap_npp = {
    .name          = "remap_npp",
    .description   = NULL_IF_CONFIG_SMALL("vstack and remap pixels using NPP."),

    .init          = remap_npp_init,
    .uninit        = remap_npp_uninit,

    .priv_size     = sizeof(RemapNPPContext),
    .priv_class    = &remap_npp_class,

    FILTER_INPUTS(remap_npp_inputs),
    FILTER_OUTPUTS(remap_npp_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_CUDA),

        .flags_internal  = FF_FILTER_FLAG_HWFRAME_AWARE,
    .flags          = AVFILTER_FLAG_HWDEVICE,
};
