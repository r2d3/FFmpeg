extern "C" {

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;

__global__ void Remap_Cuda(
    const uint8_t* top, int top_linesize,
    const uint8_t* bottom, int bottom_linesize,
    int top_w, int top_h,
    const uint16_t* xmap, int xmap_linesize,
    const uint16_t* ymap, int ymap_linesize,
    int map_w, int map_h,
    uint8_t fill_color,
    uint8_t* out, int out_linesize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 0 || x >= map_w || y < 0 || y >= map_h)
        return;
        
    int src_x = xmap[x + y * xmap_linesize/2]%(top_w);
    int src_y = ymap[x + y * ymap_linesize/2]%(2*top_h);

    unsigned char v;

    if (src_y >= 2*top_h || src_x >= top_w)
        v = fill_color;
    else if (src_y >= top_h) // source in top
        v = top[src_x + (src_y - top_h) * top_linesize];
    else
        v = bottom[src_x + src_y * bottom_linesize];
        
    out[x + y * out_linesize] = v;
}

}

