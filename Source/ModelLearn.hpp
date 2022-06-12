#ifndef __MODELLEARN_H__
#define __MODELLEARN_H__


#include "Model.hpp"


struct LayerLearnData
{
    std::size_t LayerSize;
    std::vector<float> Z;
    std::vector<float> A;
};


struct ModelLearnData
{
    std::uint8_t target;
    std::vector<LayerLearnData> LayerDatas;
};


class ModelLearn
{
public:
    ModelLearn(Model& model);

    void Fit(const Pool& learnPool, const Pool& testPool, const LearnParams& params);

private:
    friend class Model;

    void DoLearnIteration(const Pool& learnPool, const Pool& testPool, const LearnParams& params);

    ModelLearnData ApplyForLearn(const std::vector<float> input);

    std::vector<std::vector<float>> ComputeErrors(const ModelLearnData& data) const;
    std::vector<float> ComputeOutputLayerErrors(const ModelLearnData& data) const;
    std::vector<std::vector<float>> ComputePrevLayersErrors(
        const ModelLearnData& data, 
        const std::vector<float>& resultLayerErrors
    ) const;

    std::vector<ModelLayer> ComputeGradient(const ModelLearnData& data, const std::vector<std::vector<float>>& layerErrors) const;

    float GetMSELoss(const Pool& pool) const;

private:
    Model& m_model;
};


#endif