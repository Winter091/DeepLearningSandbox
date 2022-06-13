#ifndef __MODEL_H__
#define __MODEL_H__


#include <cstdint>
#include <vector>
#include <random>

#include "Pool.hpp"


enum class LossFunc : uint8_t
{
    MeanSquaredError = 0,
};


enum class ActivationFunc : uint8_t
{
    Sigmoid = 0,
};


struct LearnParams
{
    std::size_t MaxIters;
    std::size_t BatchSize;
    float DesiredLoss;
    float LearnRate;
    LossFunc lossFunc;
    ActivationFunc activationFunc;
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


class Model
{
public:
    Model();
    
    void SetInputSize(std::size_t size);
    
    void AddLayer(std::size_t size);
    const ModelLayer& GetLayer(std::size_t index) const;
    
    std::vector<float> Apply(const std::vector<float>& input) const;

    void Fit(const Pool& learnPool, const Pool& testPool, const LearnParams& params);


private:
    friend class ModelLearn;

    std::size_t m_inputSize;
    std::vector<ModelLayer> m_layers;

    std::mt19937 m_e2;
    std::uniform_real_distribution<float> m_random;


private:
    void AddLayerImpl(std::size_t size, std::size_t prevLayerSize);
};


#endif