#include "pcd_stream.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fast_lio {
namespace {
struct Field { std::string name; char type; std::size_t size,count,offset,scalar; int output; };
struct Header { std::vector<Field> fields; std::uint64_t points{0}; std::size_t stride{0},scalars{0}; std::string encoding; };
void checkCancel(const std::function<bool()> &cancel) {
  if (cancel && cancel()) throw std::runtime_error("PCD indexing canceled");
}
std::uint64_t product(std::uint64_t a,std::uint64_t b) {
  if (b && a>std::numeric_limits<std::uint64_t>::max()/b) throw std::runtime_error("PCD size overflow");
  return a*b;
}
Header readHeader(std::istream &in) {
  Header h; std::vector<std::string> names; std::vector<std::size_t> sizes,counts;
  std::vector<char> types; std::uint64_t width=0,height=1; std::size_t bytes=0;
  std::string line;
  while (std::getline(in,line)) {
    bytes+=line.size()+1;
    if (bytes>65536) throw std::runtime_error("PCD header is too large");
    std::istringstream row(line); std::string key; row>>key;
    if (key.empty() || key[0]=='#') continue;
    if (key=="FIELDS" || key=="FIELD") { std::string v; while(row>>v) names.push_back(v); }
    else if (key=="SIZE") { std::size_t v; while(row>>v) sizes.push_back(v); }
    else if (key=="COUNT") { std::size_t v; while(row>>v) counts.push_back(v); }
    else if (key=="TYPE") { char v; while(row>>v) types.push_back(v); }
    else if (key=="WIDTH") row>>width;
    else if (key=="HEIGHT") row>>height;
    else if (key=="POINTS") row>>h.points;
    else if (key=="DATA") { row>>h.encoding; break; }
  }
  if (counts.empty()) counts.assign(names.size(),1);
  if (!h.points) h.points=product(width,height);
  if (!h.points || names.empty() || names.size()>64 || sizes.size()!=names.size() ||
      types.size()!=names.size() || counts.size()!=names.size() || h.encoding.empty())
    throw std::runtime_error("missing or invalid PCD header");
  if (width && product(width,height)!=h.points) throw std::runtime_error("PCD point count disagrees with dimensions");
  const std::array<std::string,8> wanted{{"x","y","z","intensity","normal_x","normal_y","normal_z","curvature"}};
  std::array<bool,8> found{};
  for (std::size_t i=0;i<names.size();++i) {
    if (!counts[i] || counts[i]>4096 ||
        !((types[i]=='F' && (sizes[i]==4 || sizes[i]==8)) ||
          ((types[i]=='I' || types[i]=='U') && (sizes[i]==1 || sizes[i]==2 || sizes[i]==4 || sizes[i]==8))))
      throw std::runtime_error("unsupported PCD field type/size/count");
    const auto pos=std::find(wanted.begin(),wanted.end(),names[i]);
    const int slot=pos==wanted.end()?-1:static_cast<int>(pos-wanted.begin());
    if (slot>=0) {
      if (found[slot] || counts[i]!=1) throw std::runtime_error("invalid duplicate or vector-valued PCD geometry field");
      found[slot]=true;
    }
    h.fields.push_back({names[i],types[i],sizes[i],counts[i],h.stride,h.scalars,slot});
    h.stride+=sizes[i]*counts[i]; h.scalars+=counts[i];
    if (h.stride>65536) throw std::runtime_error("PCD record is too large");
  }
  if (!found[0] || !found[1] || !found[2]) throw std::runtime_error("PCD must contain x, y and z");
  product(h.points,h.stride);
  return h;
}
template<class T> float value(const char *data) { T v; std::memcpy(&v,data,sizeof(v)); return static_cast<float>(v); }
float scalar(const char *data,const Field &f) {
  if (f.type=='F') return f.size==4?value<float>(data):value<double>(data);
  if (f.type=='I') {
    switch(f.size) { case 1:return value<std::int8_t>(data); case 2:return value<std::int16_t>(data);
      case 4:return value<std::int32_t>(data); default:return value<std::int64_t>(data); }
  }
  switch(f.size) { case 1:return value<std::uint8_t>(data); case 2:return value<std::uint16_t>(data);
    case 4:return value<std::uint32_t>(data); default:return value<std::uint64_t>(data); }
}
void readExact(std::istream &in,char *data,std::size_t bytes) {
  in.read(data,static_cast<std::streamsize>(bytes));
  if (!in) throw std::runtime_error("truncated PCD data");
}
void decompressLzf(std::istream &in,const std::string &path,std::uint64_t expected,
                   const std::function<bool()> &cancel) {
  std::uint32_t sizes[2]; readExact(in,reinterpret_cast<char *>(sizes),sizeof(sizes));
  if (sizes[1]!=expected) throw std::runtime_error("compressed PCD size disagrees with header");
  std::ofstream out(path,std::ios::binary|std::ios::trunc);
  if (!out) throw std::runtime_error("cannot create PCD decompression scratch file");
  std::array<char,8192> history{}; std::array<char,65536> buffer;
  std::uint64_t consumed=0,produced=0; std::size_t buffered=0;
  auto byte=[&]() -> unsigned {
    if (consumed>=sizes[0]) throw std::runtime_error("truncated LZF token");
    const int c=in.get(); if (c<0) throw std::runtime_error("truncated LZF payload");
    ++consumed; return c;
  };
  auto emit=[&](char c) {
    if (produced>=expected) throw std::runtime_error("LZF output exceeds declared size");
    history[produced%history.size()]=c; ++produced; buffer[buffered++]=c;
    if (buffered==buffer.size()) {
      checkCancel(cancel); out.write(buffer.data(),buffered); buffered=0;
      if (!out) throw std::runtime_error("PCD scratch write failed");
    }
  };
  while (consumed<sizes[0]) {
    const auto control=byte();
    if (control<32) { for (unsigned i=0;i<=control;++i) emit(static_cast<char>(byte())); }
    else {
      unsigned length=control>>5;
      unsigned distance=(control&31)<<8;
      if (length==7) length+=byte();
      distance+=byte()+1; length+=2;
      if (distance>produced) throw std::runtime_error("invalid LZF back-reference");
      for (unsigned i=0;i<length;++i) emit(history[(produced-distance)%history.size()]);
    }
  }
  if (produced!=expected) throw std::runtime_error("LZF output is shorter than declared size");
  out.write(buffer.data(),buffered); out.close();
  if (!out) throw std::runtime_error("PCD scratch flush failed");
}
}

std::uint64_t streamPcd(const std::string &path,const std::string &scratch,
                       const std::function<void(const DiskPoint &)> &consume,
                       const std::function<bool()> &cancel) {
  std::ifstream input(path,std::ios::binary);
  if (!input) throw std::runtime_error("cannot open PCD: "+path);
  const Header h=readHeader(input); const auto body=input.tellg();
  const std::size_t batch=std::min<std::size_t>(4096,(1U<<20)/h.stride);
  if (h.encoding=="binary" || h.encoding=="binary_compressed") {
    const bool compressed=h.encoding=="binary_compressed";
    std::ifstream decompressed;
    if (compressed) {
      decompressLzf(input,scratch,product(h.points,h.stride),cancel);
      decompressed.open(scratch,std::ios::binary);
    } else {
      input.seekg(0,std::ios::end); const auto end=input.tellg();
      if (end<body || static_cast<std::uint64_t>(end-body)<product(h.points,h.stride))
        throw std::runtime_error("truncated binary PCD");
      input.seekg(body);
    }
    std::vector<char> bytes(batch*h.stride);
    std::vector<DiskPoint> points(batch);
    for (std::uint64_t first=0;first<h.points;first+=batch) {
      checkCancel(cancel);
      const auto n=std::min<std::uint64_t>(batch,h.points-first);
      if (!compressed) readExact(input,bytes.data(),n*h.stride);
      std::fill(points.begin(),points.begin()+n,DiskPoint{});
      for (const auto &f:h.fields) {
        if (f.output<0) continue;
        if (compressed) {
          decompressed.seekg(f.offset*h.points+first*f.size*f.count);
          readExact(decompressed,bytes.data(),n*f.size*f.count);
        }
        for (std::size_t i=0;i<n;++i)
          points[i][f.output]=scalar(bytes.data()+(compressed?i*f.size*f.count:i*h.stride+f.offset),f);
      }
      for (std::size_t i=0;i<n;++i) consume(points[i]);
    }
  } else if (h.encoding=="ascii") {
    std::string line;
    for (std::uint64_t i=0;i<h.points;++i) {
      if (i%4096==0) checkCancel(cancel);
      do { if (!std::getline(input,line)) throw std::runtime_error("truncated ASCII PCD"); }
      while (line.empty() || line[0]=='#');
      if (line.size()>(1U<<20)) throw std::runtime_error("ASCII PCD record is too large");
      std::istringstream row(line); std::string token; DiskPoint point{};
      for (const auto &f:h.fields) for (std::size_t j=0;j<f.count;++j) {
        if (!(row>>token)) throw std::runtime_error("incomplete ASCII PCD record");
        if (f.output>=0) {
          std::size_t parsed=0; const auto number=std::stod(token,&parsed);
          if (parsed!=token.size()) throw std::runtime_error("invalid ASCII PCD number");
          point[f.output]=static_cast<float>(number);
        }
      }
      consume(point);
    }
  } else throw std::runtime_error("unsupported PCD encoding: "+h.encoding);
  checkCancel(cancel);
  return h.points;
}
}
