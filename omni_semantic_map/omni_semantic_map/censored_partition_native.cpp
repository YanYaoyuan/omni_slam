#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

struct View {
    double total=0.;
    std::map<int,double> tokens;
};
using Hist=std::map<int,View>;

static double view_score(const View &h,int k) {
    const double alpha=1./k;
    double score=-std::lgamma(1.+h.total);
    for (const auto &item:h.tokens)
        score+=std::lgamma(alpha+item.second)-std::lgamma(alpha);
    return score;
}
static double score(const Hist &h,const int64_t *categories) {
    double s=0.;
    for (const auto &item:h) s+=view_score(item.second,categories[item.first]);
    return s;
}
static double gain(const Hist &a,const Hist &b,const int64_t *categories) {
    double s=0.;
    for (const auto &item:a) {
        const auto found=b.find(item.first);
        if (found==b.end()) continue;  // unknown view is exactly neutral
        const View &x=item.second,&y=found->second;
        const double alpha=1./categories[item.first];
        s+=std::lgamma(1.+x.total)+std::lgamma(1.+y.total)-std::lgamma(1.+x.total+y.total);
        // Noncommon categorical terms cancel algebraically.
        for (const auto &token:x.tokens) {
            const auto other=y.tokens.find(token.first);
            if (other==y.tokens.end()) continue;
            s+=std::lgamma(alpha+token.second+other->second)-std::lgamma(alpha+token.second)
               -std::lgamma(alpha+other->second)+std::lgamma(alpha);
        }
    }
    return s;
}
static void absorb(Hist &a,Hist &b) {
    for (auto &item:b) {
        auto found=a.find(item.first);
        if (found==a.end()) {a.emplace(item.first,std::move(item.second));continue;}
        found->second.total+=item.second.total;
        for (const auto &token:item.second.tokens) found->second.tokens[token.first]+=token.second;
    }
    b.clear();
}
struct Edge {
    double value;
    int a,b,va,vb;
};
struct Earlier {
    bool operator()(const Edge &a,const Edge &b) const {
        if (a.value!=b.value) return a.value<b.value;
        return std::tie(a.a,a.b)>std::tie(b.a,b.b);
    }
};

extern "C" int partition_regions(
    int n,int views,int64_t edges,const int64_t *left,const int64_t *right,
    int64_t observations,const int64_t *oa,const int64_t *ov,const int64_t *ot,
    const double *weight,const int64_t *categories,int64_t *labels,
    int64_t *merge_a,int64_t *merge_b,double *merge_values,int64_t *merge_count,
    double *metrics,char *error,int error_capacity) {
    try {
        if (n<0 || views<0 || edges<0 || observations<0) throw std::runtime_error("Negative size");
        for (int i=0;i<views;++i) if(categories[i]<2) throw std::runtime_error("K<2");
        std::vector<Hist> history(n);
        for (int64_t i=0;i<observations;++i) {
            if(oa[i]<0 || oa[i]>=n || ov[i]<0 || ov[i]>=views || ot[i]<0 || ot[i]>=categories[ov[i]]
               || !std::isfinite(weight[i]) || weight[i]<=0) throw std::runtime_error("Invalid observation");
            auto &v=history[oa[i]][ov[i]];
            v.total+=weight[i];v.tokens[ot[i]]+=weight[i];
        }
        std::vector<std::set<int>> neighbors(n);
        for (int64_t i=0;i<edges;++i) {
            if(left[i]<0 || left[i]>=right[i] || right[i]>=n) throw std::runtime_error("Invalid edge");
            if(!neighbors[left[i]].insert(right[i]).second) throw std::runtime_error("Duplicate edge");
            neighbors[right[i]].insert(left[i]);
        }
        std::vector<int> version(n,0);
        double initial=0.,gain_sum=0.;int64_t evaluated=0,merged=0;
        for(int i=0;i<n;++i) {labels[i]=i;initial+=score(history[i],categories);}
        std::priority_queue<Edge,std::vector<Edge>,Earlier> heap;
        auto offer=[&](int a,int b) {
            if(a>b) std::swap(a,b);
            double g=gain(history[a],history[b],categories);++evaluated;
            if(!std::isfinite(g)) throw std::runtime_error("Nonfinite gain");
            if(g>1e-10) heap.push({g,a,b,version[a],version[b]});
        };
        for(int64_t i=0;i<edges;++i) offer(left[i],right[i]);
        while(!heap.empty()) {
            Edge e=heap.top();heap.pop();const int a=e.a,b=e.b;
            if(labels[a]!=a || labels[b]!=b || version[a]!=e.va || version[b]!=e.vb) continue;
            const double g=gain(history[a],history[b],categories);
            if(g<=1e-10) continue;
            if(std::abs(g-e.value)>1e-8*std::max(1.,std::abs(g))) throw std::runtime_error("Stale valid gain");
            merge_a[merged]=a;merge_b[merged]=b;merge_values[merged]=g;++merged;gain_sum+=g;
            labels[b]=a;++version[a];++version[b];absorb(history[a],history[b]);
            neighbors[a].erase(b);neighbors[b].erase(a);
            for(int other:neighbors[b]) {
                neighbors[other].erase(b);neighbors[other].insert(a);neighbors[a].insert(other);
            }
            neighbors[b].clear();
            for(int other:neighbors[a]) offer(a,other);
            if(merged%5000==0) std::cerr<<"CENSORED REGION MERGES "<<merged<<" / "<<n
                <<" remaining "<<n-merged<<" evaluations "<<evaluated<<std::endl;
        }
        double final_score=0.,remaining=0.;
        for(int i=0;i<n;++i) {
            int r=i;while(labels[r]!=r) r=labels[r];labels[i]=r;
            if(r!=i) continue;
            final_score+=score(history[i],categories);
            for(int other:neighbors[i]) if(i<other)
                remaining=std::max(remaining,gain(history[i],history[other],categories));
        }
        if(remaining>1e-10) throw std::runtime_error("Positive eligible merge remains");
        if(std::abs(final_score-initial-gain_sum)>1e-9*std::max(1.,std::abs(initial)))
            throw std::runtime_error("Objective ledger mismatch");
        *merge_count=merged;
        metrics[0]=initial;metrics[1]=final_score;metrics[2]=gain_sum;metrics[3]=remaining;metrics[4]=double(evaluated);
        return 0;
    } catch(const std::exception &e) {
        if(error_capacity>0) std::snprintf(error,error_capacity,"%s",e.what());
        return 1;
    }
}
