#pragma once
#include <cmath>
#include <cstddef>

namespace fast_livo {
inline bool patch_in_frame(double u,double v,int width,int height,int half,int scale)
{
  if (!std::isfinite(u) || !std::isfinite(v) || scale<=0) return false;
  const double x=std::floor(u/scale)*scale,y=std::floor(v/scale)*scale;
  const double radius=double(half+1)*scale;
  return x-radius>=0 && y-radius>=0 && x+radius<width && y+radius<height;
}
// Constant patches carry no correlation evidence. Compute means for each pair
// rather than relying on partially initialized per-feature caches.
inline double patch_ncc(const float *a, const float *b, std::size_t size)
{
  if (!size) return 0.0;
  double ma=0,mb=0;
  for (std::size_t i=0;i<size;++i) {ma+=a[i];mb+=b[i];}
  ma/=size;mb/=size;
  double ab=0,aa=0,bb=0;
  for (std::size_t i=0;i<size;++i) {
    const double x=a[i]-ma,y=b[i]-mb;
    ab+=x*y;aa+=x*x;bb+=y*y;
  }
  return aa>1e-12 && bb>1e-12 ? ab/std::sqrt(aa*bb) : 0.0;
}
}
