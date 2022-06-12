#ifndef __MODELLEARN_H__
#define __MODELLEARN_H__


#include "Model.hpp"


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
};


template <typename T>
using Matrix = std::vector<std::vector<T>>;


class ModelLearn
{
public:
    ModelLearn(Model& model);

    void Fit(const Pool& learnPool, const Pool& testPool, const LearnParams& params);

private:
    friend class Model;

    void DoLearnIteration(const Pool& learnPool, const Pool& testPool, const LearnParams& params);

    ModelActivations ComputeActivations(const std::vector<float> input);

    Matrix<float> ComputeErrors(
        const ModelActivations& activations, uint8_t target) const;

    std::vector<float> ComputeOutputLayerErrors(
        const ModelActivations& activations, uint8_t target) const;

    ModelGradient ComputeGradient(
        const ModelActivations& activations, const Matrix<float>& layerErrors) const;

    float GetMSELoss(const Pool& pool) const;

private:
    Model& m_model;
};


#endif