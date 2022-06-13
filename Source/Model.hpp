#ifndef __MODEL_H__
#define __MODEL_H__


#include <cstdint>
#include <vector>
#include <random>
#include <functional>

#include "Pool.hpp"


enum class LossFunc : uint8_t
{
    SquaredError = 0,
    LogLoss,
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

    void SetLossFunc(LossFunc lossFunc);
    void SetActivationFunc(ActivationFunc activationFunc);
    
    std::vector<float> Apply(const std::vector<float>& input) const;

    void SetupParams(const LearnParams& params);

    void Fit(const Pool& learnPool, const Pool& testPool, const LearnParams& params);
    float GetPoolLoss(const Pool& pool) const;

private:
    friend class ModelLearn;

    std::size_t m_inputSize;
    std::vector<ModelLayer> m_layers;

    std::mt19937 m_e2;
    std::uniform_real_distribution<float> m_random;

    std::function<float(const std::vector<float>&, const std::vector<float>)> m_lossFunc;
    std::function<float(float, float)> m_lossFuncDerivative;

    std::function<float(float)> m_activationFunc;
    std::function<float(float)> m_activationFuncDerivative;

private:
    void AddLayerImpl(std::size_t size, std::size_t prevLayerSize);
};


#endif