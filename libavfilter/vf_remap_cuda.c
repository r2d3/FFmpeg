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
#include "filters.h"
#include "framesync.h"
#include "internal.h"
#include "video.h"

#include "cuda/load_helper.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(avctx, ctx->hwctx->internal->cuda_dl, x)

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_NONE
};

static const enum AVPixelFormat supported_map_formats[] = {
    AV_PIX_FMT_YUV444P16LE,
    AV_PIX_FMT_GRAY16LE,
    AV_PIX_FMT_NONE
};

typedef struct RemapCUDAContext {
    const AVClass* class;
    
    enum AVPixelFormat in_format_top;
    enum AVPixelFormat in_format_bottom;
    
    AVCUDADeviceContext *hwctx;

    CUmodule    cu_module;
    CUfunction  cu_func;
    CUstream    cu_stream;
    
    AVBufferRef* frames_ctx;
    AVFrame* vstack_frame;

    FFFrameSync fs;
} RemapCUDAContext;

#define OFFSET(x) offsetof(RemapCUDAContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption remap_cuda_options[] = {
    { NULL }
};

FRAMESYNC_DEFINE_CLASS(remap_cuda, RemapCUDAContext, fs);

static int format_is_supported(const enum AVPixelFormat formats[], enum AVPixelFormat fmt)
{
    for (int i = 0; formats[i] != AV_PIX_FMT_NONE; i++)
        if (formats[i] == fmt)
            return 1;
    return 0;
}

static int remap_cuda(FFFrameSync *fs)
{
    int ret;
    
    AVFilterContext *avctx = fs->parent;
    RemapCUDAContext *ctx = avctx->priv;
    CudaFunctions *cu = ctx->hwctx->internal->cuda_dl;
    AVFilterLink *outlink = avctx->outputs[0];
    AVFilterLink *inlink = avctx->inputs[0];
    AVFrame *input_top, *input_bottom;
    CUcontext dummy;
    CUDA_MEMCPY2D cpy = { 0 };
    
    // read top and bottom frames from inputs
    ret = ff_framesync_dualinput_get(fs, &input_top, &input_bottom);
    if (ret < 0)
        return ret;
    
    if (!input_top)
        return AVERROR_BUG;

    if (!input_bottom)
    {
        av_log(avctx, AV_LOG_ERROR, "No bottom stream!!!\n");
        return ff_filter_frame(outlink, input_top);
    }
    
    ret = ff_inlink_make_frame_writable(inlink, &input_top);
    if (ret < 0) {
        av_frame_free(&input_top);
        return ret;
    }
    
#if 0
    av_log(avctx, AV_LOG_ERROR, "Top: %d %d %d %d %d\n",
           input_top->width, input_top->height, input_top->linesize[0], input_top->linesize[1],
           input_top->linesize[2]);
    av_log(avctx, AV_LOG_ERROR, "Bottom: %d %d %d %d %d\n",
           input_bottom->width, input_bottom->height, input_bottom->linesize[0], input_bottom->linesize[1],
           input_bottom->linesize[2]);
#endif
    
    ret = CHECK_CU(cu->cuCtxPushCurrent(ctx->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;
    
    cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy.srcDevice = (CUdeviceptr)input_bottom->data[0];
    cpy.dstDevice = (CUdeviceptr)input_top->data[0];
    cpy.srcPitch = input_bottom->linesize[0];
    cpy.dstPitch = input_top->linesize[0];
    cpy.WidthInBytes = input_top->linesize[0];
    cpy.Height = input_top->height;

    ret = CHECK_CU(cu->cuMemcpy2DAsync(&cpy, ctx->cu_stream));
    if (ret < 0)
        return ret;
    
    switch(ctx->in_format_top)
    {
        case AV_PIX_FMT_YUV420P:
        {
            cpy.srcDevice = (CUdeviceptr)input_bottom->data[1];
            cpy.dstDevice = (CUdeviceptr)input_top->data[1];
            cpy.srcPitch = input_bottom->linesize[1];
            cpy.dstPitch = input_top->linesize[1];
            cpy.WidthInBytes = input_top->linesize[1];
            cpy.Height = input_top->height / 2;

            ret = CHECK_CU(cu->cuMemcpy2DAsync(&cpy, ctx->cu_stream));
            if (ret < 0)
                return ret;

            cpy.srcDevice = (CUdeviceptr)input_bottom->data[2];
            cpy.dstDevice = (CUdeviceptr)input_top->data[2];
            cpy.srcPitch = input_bottom->linesize[2];
            cpy.dstPitch = input_top->linesize[2];
            cpy.WidthInBytes = input_top->linesize[2];
            cpy.Height = input_top->height / 2;

            ret = CHECK_CU(cu->cuMemcpy2DAsync(&cpy, ctx->cu_stream));
            if (ret < 0)
                return ret;
        }
            break;
        default:
            av_log(avctx, AV_LOG_ERROR, "Passed unsupported top pixel format\n");
            av_frame_free(&input_top);
            CHECK_CU(cu->cuCtxPopCurrent(&dummy));
            return AVERROR_BUG;
    }
    
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        return ret;

    // TODO: do real processing, for the moment, just copy top
    return ff_filter_frame(outlink, input_top);
}

static int remap_cuda_init(AVFilterContext* avctx)
{
    RemapCUDAContext* ctx = avctx->priv;
    av_log(avctx, AV_LOG_DEBUG, "remap_cuda_init\n");

    ctx->vstack_frame = av_frame_alloc();
    if (!ctx->vstack_frame)
        return AVERROR(ENOMEM);

    return 0;
}

static void remap_cuda_uninit(AVFilterContext* avctx)
{
    RemapCUDAContext* ctx = avctx->priv;
    av_log(avctx, AV_LOG_DEBUG, "remap_cuda_uninit\n");

    ff_framesync_uninit(&ctx->fs);
    
    if (ctx->hwctx && ctx->cu_module) {
        CudaFunctions *cu = ctx->hwctx->internal->cuda_dl;
        CUcontext dummy;

        CHECK_CU(cu->cuCtxPushCurrent(ctx->hwctx->cuda_ctx));
        CHECK_CU(cu->cuModuleUnload(ctx->cu_module));
        ctx->cu_module = NULL;
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }

    av_buffer_unref(&ctx->frames_ctx);
    av_frame_free(&ctx->vstack_frame);
}

static int remap_cuda_activate(AVFilterContext *avctx)
{
    RemapCUDAContext *ctx = avctx->priv;
    av_log(ctx, AV_LOG_DEBUG, "remap_cuda_activate\n");

    return ff_framesync_activate(&ctx->fs);
}

static int config_props(AVFilterLink *inlink)
{
    AVHWFramesContext *input_frames;
    AVFilterContext *ctx = inlink->dst;
    RemapCUDAContext *s = ctx->priv;
    AVHWFramesContext     *hw_frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = hw_frames_ctx->device_ctx->hwctx;

    av_log(ctx, AV_LOG_DEBUG, "config_props\n");

    if (!inlink->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "CUDA filtering requires a "
               "hardware frames context on the input.\n");
        return AVERROR(EINVAL);
    }

    input_frames = (AVHWFramesContext *)inlink->hw_frames_ctx->data;
    if (input_frames->format != AV_PIX_FMT_CUDA)
    {
        av_log(ctx, AV_LOG_ERROR, "Input frame is not CUDA.\n");
        return AVERROR(EINVAL);
    }
    
    s->hwctx = device_hwctx;
    s->cu_stream = s->hwctx->stream;

    /* Extract the device and default output format from the first input. */
    if (ctx->inputs[0] != inlink)
        return 0;

    return 0;
}

static int config_map(AVFilterLink *inlink)
{
    AVHWFramesContext *input_frames;
    AVFilterContext *ctx = inlink->dst;

    av_log(ctx, AV_LOG_DEBUG, "config_map\n");
    
    if (!inlink->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "remap_cuda requires a "
               "hardware frames context on the input.\n");
        return AVERROR(EINVAL);
    }

    if (!inlink->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "CUDA filtering requires a "
               "hardware frames context on the input.\n");
        return AVERROR(EINVAL);
    }

    input_frames = (AVHWFramesContext *)inlink->hw_frames_ctx->data;
    if (input_frames->format != AV_PIX_FMT_CUDA)
    {
        av_log(ctx, AV_LOG_ERROR, "Input map is not CUDA.\n");
        return AVERROR(EINVAL);
    }
    
    return 0;
}

static int config_output(AVFilterLink* outlink)
{
    extern const unsigned char ff_vf_remap_cuda_ptx_data[];
    extern const unsigned int ff_vf_remap_cuda_ptx_len;

    int ret;
    AVFilterContext* avctx = outlink->src;
    AVHWFramesContext *out_ctx, *in_ctx;
    RemapCUDAContext* ctx = avctx->priv;
    AVFilterLink *inlink_top = avctx->inputs[0];
    AVFilterLink *inlink_bot = avctx->inputs[1];
    AVFilterLink *inlink_xmap = avctx->inputs[2];
    AVFilterLink *inlink_ymap = avctx->inputs[3];
    AVHWFramesContext *frames_ctx_top, *frames_ctx_bot, *frames_ctx_xmap, *frames_ctx_ymap;
    FFFrameSyncIn *in;

    CUcontext dummy;
    CudaFunctions *cu;

    av_log(avctx, AV_LOG_DEBUG, "config_output\n");
    
    if (!inlink_top || !inlink_top->hw_frames_ctx || !inlink_top->hw_frames_ctx->data)
        return AVERROR(EINVAL);
    frames_ctx_top = (AVHWFramesContext*)inlink_top->hw_frames_ctx->data;
    
    if (!inlink_bot || !inlink_bot->hw_frames_ctx || !inlink_bot->hw_frames_ctx->data)
        return AVERROR(EINVAL);
    frames_ctx_bot = (AVHWFramesContext*)inlink_bot->hw_frames_ctx->data;
    
    if (!inlink_xmap || !inlink_xmap->hw_frames_ctx || !inlink_xmap->hw_frames_ctx->data)
        return AVERROR(EINVAL);
    frames_ctx_xmap = (AVHWFramesContext*)inlink_xmap->hw_frames_ctx->data;

    if (!inlink_ymap || !inlink_ymap->hw_frames_ctx || !inlink_ymap->hw_frames_ctx->data)
        return AVERROR(EINVAL);
    frames_ctx_ymap = (AVHWFramesContext*)inlink_ymap->hw_frames_ctx->data;

    // check top input formats
    
    if (!frames_ctx_top) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on top input\n");
        return AVERROR(EINVAL);
    }
    
    ctx->in_format_top = frames_ctx_top->sw_format;
    if (!format_is_supported(supported_formats, ctx->in_format_top)) {
        av_log(avctx, AV_LOG_ERROR, "Unsupported top input format: %s\n",
               av_get_pix_fmt_name(ctx->in_format_top));
        return AVERROR(ENOSYS);
    }
    
    // check bottom input formats
    
    if (!frames_ctx_bot) {
        av_log(avctx, AV_LOG_ERROR, "No hw context provided on bottom input\n");
        return AVERROR(EINVAL);
    }
    
    ctx->in_format_bottom = frames_ctx_bot->sw_format;
    if (!format_is_supported(supported_formats, ctx->in_format_bottom)) {
        av_log(avctx, AV_LOG_ERROR, "Unsupported bottom input format: %s\n",
               av_get_pix_fmt_name(ctx->in_format_bottom));
        return AVERROR(ENOSYS);
    }
    
    // check xmap/ymap

    if (!frames_ctx_xmap) {
        av_log(avctx, AV_LOG_ERROR, "No hw context provided on xmap input\n");
        return AVERROR(EINVAL);
    }
    if (!format_is_supported(supported_map_formats, frames_ctx_xmap->sw_format)) {
        av_log(avctx, AV_LOG_ERROR, "Unsupported xmap input format: %s\n",
               av_get_pix_fmt_name(frames_ctx_xmap->sw_format));
        return AVERROR(ENOSYS);
    }

    if (!frames_ctx_ymap) {
        av_log(avctx, AV_LOG_ERROR, "No hw context provided on ymap input\n");
        return AVERROR(EINVAL);
    }
    if (!format_is_supported(supported_map_formats, frames_ctx_xmap->sw_format)) {
        av_log(avctx, AV_LOG_ERROR, "Unsupported ymap input format: %s\n",
               av_get_pix_fmt_name(frames_ctx_ymap->sw_format));
        return AVERROR(ENOSYS);
    }

    if (inlink_xmap->w != inlink_ymap->w || inlink_xmap->h != inlink_ymap->h) {
        av_log(avctx, AV_LOG_ERROR, "Third input link %s parameters "
               "(size %dx%d) do not match the corresponding "
               "fourth input link %s parameters (%dx%d)\n",
               avctx->input_pads[2].name, inlink_xmap->w, inlink_xmap->h,
               avctx->input_pads[3].name, inlink_ymap->w, inlink_ymap->h);
        return AVERROR(EINVAL);
    }

    // check we can vstack pictures with those pixel formats
    
    if (ctx->in_format_top != ctx->in_format_bottom) {
        av_log(avctx, AV_LOG_ERROR, "Can't vstack %s on %s \n",
               av_get_pix_fmt_name(ctx->in_format_top), av_get_pix_fmt_name(ctx->in_format_bottom));
        return AVERROR(EINVAL);
    }
    
    // Allocate vstack image
    in_ctx = (AVHWFramesContext*)avctx->inputs[0]->hw_frames_ctx->data;

    av_log(avctx, AV_LOG_DEBUG, "config_output: av_hwframe_ctx_alloc\n");
    ctx->frames_ctx = av_hwframe_ctx_alloc(in_ctx->device_ref);
    if (!ctx->frames_ctx)
        goto fail;

    out_ctx = (AVHWFramesContext*)ctx->frames_ctx->data;
    out_ctx->format = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = in_ctx->sw_format;
    out_ctx->width = FFALIGN(inlink_top->w, 32);
    out_ctx->height = FFALIGN(inlink_top->h+inlink_bot->h, 32);
    
    av_log(avctx, AV_LOG_DEBUG, "config_output: av_hwframe_ctx_init\n");
    ret = av_hwframe_ctx_init(ctx->frames_ctx);
    if (ret < 0)
        goto fail;

    av_log(avctx, AV_LOG_DEBUG, "config_output: av_hwframe_get_buffer\n");
    av_frame_unref(ctx->vstack_frame);
    ret = av_hwframe_get_buffer(ctx->frames_ctx, ctx->vstack_frame, 0);
    if (ret < 0)
        goto fail;

    // TODO: for the moment vstack is our output buffer
    av_log(avctx, AV_LOG_DEBUG, "config_output: av_buffer_ref\n");
    avctx->outputs[0]->hw_frames_ctx = av_buffer_ref(ctx->frames_ctx);
    if (!avctx->outputs[0]->hw_frames_ctx)
        goto fail;

    // load functions

    cu = ctx->hwctx->internal->cuda_dl;

    ret = CHECK_CU(cu->cuCtxPushCurrent(ctx->hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;

    ret = ff_cuda_load_module(avctx, ctx->hwctx, &ctx->cu_module,
                              ff_vf_remap_cuda_ptx_data, ff_vf_remap_cuda_ptx_len);
    if (ret < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return ret;
    }

    ret = CHECK_CU(cu->cuModuleGetFunction(&ctx->cu_func, ctx->cu_module, "Remap_Cuda"));
    if (ret < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return ret;
    }

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    
    // init quad input
    
    ret = ff_framesync_init(&ctx->fs, avctx, 4);
    if (ret < 0) {
        av_buffer_unref(&ctx->frames_ctx);
        return ret;
    }
    
    in = ctx->fs.in;
    in[0].time_base = inlink_top->time_base;
    in[1].time_base = inlink_bot->time_base;
    in[2].time_base = inlink_xmap->time_base;
    in[3].time_base = inlink_ymap->time_base;
    in[0].sync   = 2;
    in[0].before = EXT_STOP;
    in[0].after  = EXT_STOP;
    in[1].sync   = 2;
    in[1].before = EXT_STOP;
    in[1].after  = EXT_STOP;
    in[2].sync   = 1;
    in[2].before = EXT_NULL;
    in[2].after  = EXT_INFINITY;
    in[3].sync   = 1;
    in[3].before = EXT_NULL;
    in[3].after  = EXT_INFINITY;
    ctx->fs.opaque   = ctx;
    ctx->fs.on_event = &remap_cuda;

    ret = ff_framesync_configure(&ctx->fs);
    if (ret < 0) {
        av_buffer_unref(&ctx->frames_ctx);
        return ret;
    }
    
    return 0;

fail:
    av_buffer_unref(&ctx->frames_ctx);
    return AVERROR(ENOMEM);
}

static const AVFilterPad remap_cuda_inputs[] = {
    {
        .name         = "top",
        .type         = AVMEDIA_TYPE_VIDEO,
    },
    {
        .name         = "bottom",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_props,
    },
    {
        .name         = "xmap",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_map,
    },
    {
        .name         = "ymap",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_map,
    },
};

static const AVFilterPad remap_cuda_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
};

const AVFilter ff_vf_remap_cuda = {
    .name          = "remap_cuda",
    .description   = NULL_IF_CONFIG_SMALL("vstack and remap pixels using CUDA."),

    .init          = &remap_cuda_init,
    .uninit        = &remap_cuda_uninit,
    .activate      = &remap_cuda_activate,
    .preinit       = remap_cuda_framesync_preinit,

    .priv_size     = sizeof(RemapCUDAContext),
    .priv_class    = &remap_cuda_class,

    FILTER_INPUTS(remap_cuda_inputs),
    FILTER_OUTPUTS(remap_cuda_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_CUDA),

    .flags_internal  = FF_FILTER_FLAG_HWFRAME_AWARE,
    .flags          = AVFILTER_FLAG_HWDEVICE,
};
