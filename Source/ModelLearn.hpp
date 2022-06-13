#ifndef __MODELLEARN_H__
#define __MODELLEARN_H__


#include "Model.hpp"
#include <algorithm>


template <typename T>
using Matrix = std::vector<std::vector<T>>;


struct LayerActivations
{
    std::size_t LayerSize;
    std::vector<float> PreActivations;
    std::vector<float> Activations;
};


struct ModelActivations
{
    std::vector<LayerActivations> Layers;
};


struct ModelGradient
{
    std::vector<ModelLayer> Layers;

    ModelGradient& operator+=(const ModelGradient& other)
    {
        for (int layerIndex = 0; layerIndex < Layers.size(); layerIndex++) {
            auto& thisLayer = Layers[layerIndex];
            auto& otherLayer = other.Layers[layerIndex];

            for (int i = 0; i < thisLayer.Weights.size(); i++) {
                thisLayer.Weights[i] += otherLayer.Weights[i];
            }

            for (int i = 0; i < thisLayer.Biases.size(); i++) {
                thisLayer.Biases[i] += otherLayer.Biases[i];
            }
        }

        return *this;
    }

    ModelGradient& operator/=(float scalar)
    {
        for (int layerIndex = 0; layerIndex < Layers.size(); layerIndex++) {
            auto& layer = Layers[layerIndex];

            for (int i = 0; i < layer.Biases.size(); i++) {
                layer.Biases[i] /= scalar;
            }

            for (int i = 0; i < layer.Weights.size(); i++) {
                layer.Weights[i] /= scalar;
            }
        }

        return *this;
    }
};


class ModelLearn
{
public:
    ModelLearn(Model& model);

    void Fit(const Pool& learnPool, const Pool& testPool, const LearnParams& params);

private:
    friend class Model;

    void SetupParams(const LearnParams& params);

    void DoLearnIteration(const Pool& learnPool, const Pool& testPool);

    ModelActivations ComputeActivations(const std::vector<float>& input) const;

    Matrix<float> ComputeErrors(
        const ModelActivations& activations, uint8_t target) const;

    std::vector<float> ComputeOutputLayerErrors(
        const ModelActivations& activations, uint8_t target) const;

    ModelGradient ComputeGradient(
        const ModelActivations& activations, const Matrix<float>& layerErrors) const;

    void ApplyGradient(const ModelGradient& gradient);

    float GetPoolLoss(const Pool& pool) const;

private:
    Model& m_model;

    LearnParams m_learnParams;

    std::function<float(const std::vector<float>&, const std::vector<float>)> m_lossFunc;
    std::function<float(float, float)> m_lossFuncDerivative;

    std::function<float(float)> m_activationFunc;
    std::function<float(float)> m_activationFuncDerivative;
};


#endif