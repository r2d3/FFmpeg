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
#include "libavutil/colorspace.h"
#include "libavutil/pixdesc.h"
#include "libavutil/opt.h"

#include "avfilter.h"
#include "filters.h"
#include "framesync.h"
#include "internal.h"
#include "video.h"

#include "cuda/load_helper.h"

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

#define BLOCK_X 32
#define BLOCK_Y 16

#define CHECK_CU(x) FF_CUDA_CHECK_DL(avctx, ctx->hwctx->internal->cuda_dl, x)

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV444P,
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

/**
 * Call remap kernell for a plane
 */
static int remap_cuda_call_kernel(
    AVFilterContext *avctx,
    uint8_t* top_data, int top_linesize,
    uint8_t* bottom_data, int bottom_linesize,
    int top_width, int top_height,
    uint16_t* xmap_data, int xmap_linesize,
    uint16_t* ymap_data, int ymap_linesize,
    int xmap_width, int xmap_height,
    uint8_t fill_color,
    uint8_t* output, int output_linesize)
{
    RemapCUDAContext *ctx = avctx->priv;
    CudaFunctions *cu = ctx->hwctx->internal->cuda_dl;

    void* kernel_args[] = {
        &top_data, &top_linesize,
        &bottom_data, &bottom_linesize,
        &top_width, &top_height,
        &xmap_data, &xmap_linesize,
        &ymap_data, &ymap_linesize,
        &xmap_width, &xmap_height,
        &fill_color,
        &output, &output_linesize
    };

    return CHECK_CU(cu->cuLaunchKernel(
        ctx->cu_func,
        DIV_UP(xmap_width, BLOCK_X), DIV_UP(xmap_height, BLOCK_Y), 1,
        BLOCK_X, BLOCK_Y, 1,
        0, ctx->cu_stream, kernel_args, NULL));
}


static int remap_cuda(FFFrameSync *fs)
{
    int i, ret;
    
    AVFilterContext *avctx = fs->parent;
    RemapCUDAContext *ctx = avctx->priv;
    CudaFunctions *cu = ctx->hwctx->internal->cuda_dl;
    AVFilterLink *outlink = avctx->outputs[0];
    AVFrame *input_top, *input_bottom, *input_xmap, *input_ymap;
    AVFrame *output;
    CUcontext dummy;
    unsigned char black[3] = {
        RGB_TO_Y_BT709(0, 0, 0),
        RGB_TO_U_BT709(0, 0, 0, 0),
        RGB_TO_V_BT709(0, 0, 0, 0)
    };
    
    av_log(avctx, AV_LOG_DEBUG, "remap_cuda: %d %d\n", outlink->w, outlink->h);

    // read top and bottom frames from inputs
    ret = ff_framesync_dualinput_get(fs, &input_top, &input_bottom);
    if (ret < 0)
        return ret;
    ret = ff_framesync_get_frame(fs, 2, &input_xmap, 0);
    if (ret < 0)
        return ret;
    ret = ff_framesync_get_frame(fs, 3, &input_ymap, 0);
    if (ret < 0)
        return ret;
    
    if (!input_top || !input_bottom || !input_xmap || !input_ymap)
        return AVERROR_BUG;
    
    av_log(avctx, AV_LOG_DEBUG, "Top: %d %d %d %d %d\n",
           input_top->width, input_top->height, input_top->linesize[0], input_top->linesize[1],
           input_top->linesize[2]);
    av_log(avctx, AV_LOG_DEBUG, "Bottom: %d %d %d %d %d\n",
           input_bottom->width, input_bottom->height, input_bottom->linesize[0], input_bottom->linesize[1],
           input_bottom->linesize[2]);
    av_log(avctx, AV_LOG_DEBUG, "xmap: %d %d %d %d %d\n",
           input_xmap->width, input_xmap->height, input_xmap->linesize[0], input_xmap->linesize[1],
           input_xmap->linesize[2]);
    av_log(avctx, AV_LOG_DEBUG, "ymap: %d %d %d %d %d\n",
           input_ymap->width, input_ymap->height, input_ymap->linesize[0], input_ymap->linesize[1],
           input_ymap->linesize[2]);

    output = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!output) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }
    
    ret = CHECK_CU(cu->cuCtxPushCurrent(ctx->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;
    
    // Do the processing
    for (i = 0; i < 3; i++)
    {
        remap_cuda_call_kernel(avctx,
                               input_top->data[i], input_top->linesize[i],
                               input_bottom->data[i], input_bottom->linesize[i],
                               input_top->width, input_top->height,
                               (uint16_t*)input_xmap->data[0], input_xmap->linesize[0],
                               (uint16_t*)input_ymap->data[0], input_ymap->linesize[0],
                               input_xmap->width, input_xmap->height,
                               black[i],
                               output->data[i], output->linesize[i]
                               );
    }

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        return ret;

    ret = av_frame_copy_props(output, input_top);

    av_log(avctx, AV_LOG_DEBUG, "Filter output: %s, %ux%u (%"PRId64").\n",
           av_get_pix_fmt_name(output->format),
           output->width, output->height, output->pts);

    return ff_filter_frame(outlink, output);

fail:
    av_frame_free(&output);
    return ret;
}

static int remap_cuda_init(AVFilterContext* avctx)
{
    av_log(avctx, AV_LOG_DEBUG, "remap_cuda_init\n");

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
    RemapCUDAContext* ctx = avctx->priv;
    AVFilterLink *inlink_top = avctx->inputs[0];
    AVFilterLink *inlink_bot = avctx->inputs[1];
    AVFilterLink *inlink_xmap = avctx->inputs[2];
    AVFilterLink *inlink_ymap = avctx->inputs[3];
    AVHWFramesContext *frames_ctx_top, *frames_ctx_bot, *frames_ctx_xmap, *frames_ctx_ymap;
    FFFrameSyncIn *in;
    AVBufferRef *output_frames_ref = NULL;
    AVHWFramesContext *output_frames;

    CUcontext dummy;
    CudaFunctions *cu;

    av_log(avctx, AV_LOG_DEBUG, "config_output\n");
    
    {
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
    }

    {
        outlink->w = inlink_xmap->w;
        outlink->h = inlink_xmap->h;
        outlink->sample_aspect_ratio = inlink_top->sample_aspect_ratio;
        outlink->frame_rate = inlink_top->frame_rate;

        output_frames_ref = av_hwframe_ctx_alloc(frames_ctx_top->device_ref);
        if (!output_frames_ref) {
            ret = AVERROR(ENOMEM);
            av_buffer_unref(&output_frames_ref);
            return ret;
        }
        output_frames = (AVHWFramesContext*)output_frames_ref->data;

        output_frames->format    = AV_PIX_FMT_CUDA;
        output_frames->sw_format = frames_ctx_top->sw_format;
        output_frames->width     = outlink->w;
        output_frames->height    = outlink->h;

        ret = av_hwframe_ctx_init(output_frames_ref);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "Failed to initialise output "
                   "frames: %d.\n", ret);
            av_buffer_unref(&output_frames_ref);
            return ret;
        }

        outlink->hw_frames_ctx = output_frames_ref;
    }

    if (ctx->in_format_top != ctx->in_format_bottom) {
        av_log(avctx, AV_LOG_ERROR, "Can't vstack %s on %s \n",
               av_get_pix_fmt_name(ctx->in_format_top), av_get_pix_fmt_name(ctx->in_format_bottom));
        return AVERROR(EINVAL);
    }
    
    // load functions

    {
        cu = ctx->hwctx->internal->cuda_dl;

        ret = CHECK_CU(cu->cuCtxPushCurrent(ctx->hwctx->cuda_ctx));
        if (ret < 0) {
            CHECK_CU(cu->cuCtxPopCurrent(&dummy));
            return ret;
        }

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
    }

    // init quad input

    {
        ret = ff_framesync_init(&ctx->fs, avctx, 4);
        if (ret < 0)
            return ret;

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
        if (ret < 0)
            return ret;
    }

    return 0;
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
