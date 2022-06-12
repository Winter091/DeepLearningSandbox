#ifndef __MODELVISUALIZER_H__
#define __MODELVISUALIZER_H__

#include "Model.hpp"
#include <string>


class ModelVisualizer
{
public:
    static void DumpLayerToBmps(const ModelLayer& layer, const std::string& basePath);

private:
    static void DumpElemToBmp(
        const ModelLayer& layer, std::size_t layerElemIndex, const std::string& path);
};



#endif