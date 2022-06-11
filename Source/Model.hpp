#ifndef __MODEL_H__
#define __MODEL_H__


#include <cstdint>
#include <vector>
#include <random>

#include "Pool.hpp"


struct LearnParams
{
    std::size_t NumIters;
    std::size_t BatchSize;
};


struct ModelLayer
{
    std::size_t Size;
    std::size_t PrevLayerSize;
    std::vector<float> Weights;
    std::vector<float> Biases;

    ModelLayer(std::size_t size, std::size_t prevLayerSize)
        : Size(size)
        , PrevLayerSize(prevLayerSize)
        , Weights(prevLayerSize * size, 0.0f)
        , Biases(size, 0.0f)
    {}

    float& WeightAt(std::size_t i, std::size_t j) { return Weights[i * Size + j]; }
    const float& WeightAt(std::size_t i, std::size_t j) const { return Weights[i * Size + j]; }
};


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


class Model
{
public:
    Model();
    
    void SetInputSize(std::size_t size);
    
    void AddLayer(std::size_t size);
    const ModelLayer& GetLayer(std::size_t index) const;
    
    std::vector<float> Apply(const std::vector<float>& input) const;

    void Fit(const Pool& learnPool, const Pool& testPool, const LearnParams& params);

    float GetMSELoss(const Pool& pool) const;


private:
    std::size_t m_inputSize;
    std::vector<ModelLayer> m_layers;

    std::mt19937 m_e2;
    std::uniform_real_distribution<float> m_random;


private:
    void AddLayerImpl(std::size_t size, std::size_t prevLayerSize);

    void DoLearnIteration(const Pool& learnPool, const Pool& testPool, const LearnParams& params);
    ModelLearnData ApplyForLearn(const std::vector<float> input);
    std::vector<std::vector<float>> ComputeErrors(const ModelLearnData& data) const;

    std::vector<float> ComputeOutputLayerErrors(const ModelLearnData& data) const;
    std::vector<std::vector<float>> ComputePrevLayersErrors(
        const ModelLearnData& data, 
        const std::vector<float>& resultLayerErrors
    ) const;

    std::vector<ModelLayer> ComputeGradient(const ModelLearnData& data, const std::vector<std::vector<float>>& layerErrors) const;
};


#endif