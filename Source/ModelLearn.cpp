#include "ModelLearn.hpp"


ModelLearn::ModelLearn(Model& model)
    : m_model(model)
{}


void ModelLearn::Fit(const Pool& learnPool, const Pool& testPool, const LearnParams& params)
{
    m_learnParams = params;
    
    printf("Iteration %3d: loss = %.3f\n", 0, m_model.GetPoolLoss(testPool));

    for (int i = 1; i <= m_learnParams.MaxIters; i++) {
        DoLearnIteration(learnPool, testPool);

        if (i % 25 == 0) {
            float loss = m_model.GetPoolLoss(testPool);
            printf("Iteration %3d: loss = %.3f\n", i, loss);
            if (loss < params.DesiredLoss) {
                break;
            }
        }
    }
}


void ModelLearn::DoLearnIteration(const Pool& learnPool, const Pool& testPool)
{
    Pool batchLearnPool = learnPool.TakeRandom(m_learnParams.BatchSize);
    ModelGradient avgGradient;

    for (int i = 0; i < batchLearnPool.GetElements().size(); i++) {
        const auto& element = batchLearnPool.GetElements()[i];

        ModelActivations activations = ComputeActivations(element.Features);
        Matrix<float> errors = ComputeErrors(activations, element.Target);
        ModelGradient gradient = ComputeGradient(activations, errors);

        if (i == 0) {
            avgGradient = std::move(gradient);
        } else {
            avgGradient += gradient;
        }
    }

    avgGradient /= batchLearnPool.GetSize();
    ApplyGradient(avgGradient);
}


Matrix<float> ModelLearn::ComputeErrors(
    const ModelActivations& activations, uint8_t target) const
{
    Matrix<float> result;
    result.reserve(m_model.m_layers.size());
    result.push_back(ComputeOutputLayerErrors(activations, target));

    for (int i = m_model.m_layers.size() - 2; i >= 0; i--) {
        auto* nextLayerErrors = &result.back();
        auto& nextLayer = m_model.m_layers[i + 1];

        std::vector<float> currLayerErrors(m_model.m_layers[i].Size);
        for (int j = 0; j < currLayerErrors.size(); j++) {

            float sum = 0.0f;
            for (int k = 0; k < nextLayerErrors->size(); k++) {
                sum += (nextLayer.WeightAt(j, k) * nextLayerErrors->at(k));
            }

            float z = activations.Layers[i + 1].PreActivations[j];
            currLayerErrors[j] = m_model.m_activationFuncDerivative(z) * sum;
        }

        result.push_back(std::move(currLayerErrors));
    }

    std::reverse(result.begin(), result.end());
    return result;
}


std::vector<float> ModelLearn::ComputeOutputLayerErrors(
    const ModelActivations& data, uint8_t target) const
{
    std::vector<float> errors(m_model.m_layers.back().Size, 0.0f);

    std::vector<float> correctSolution(errors.size(), 0.0f);
    correctSolution[target] = 1.0f;

    for (int i = 0; i < errors.size(); i++) {
        float a = data.Layers.back().Activations[i];
        float y = correctSolution[i];
        float z = data.Layers.back().PreActivations[i];
        errors[i] = m_model.m_lossFuncDerivative(a, y) * m_model.m_activationFuncDerivative(z);
    }

    return errors;
}


ModelActivations ModelLearn::ComputeActivations(const std::vector<float>& input) const
{
    if (input.size() != m_model.m_inputSize) {
        fprintf(stderr, "Input size (%d) should be %d\n", input.size(), m_model.m_inputSize);
        exit(EXIT_FAILURE);
    }

    if (m_model.m_layers.empty()) {
        fprintf(stderr, "Modes has no layers");
        exit(EXIT_FAILURE);
    }
    
    ModelActivations activations;
    activations.Layers.reserve(m_model.m_layers.size() + 1);

    LayerActivations inputLayerActivations;
    inputLayerActivations.LayerSize = input.size();
    inputLayerActivations.Activations = input;
    activations.Layers.push_back(std::move(inputLayerActivations));

    std::vector<float> prevLayerActivations = input;

    for (const auto& layer : m_model.m_layers) {
        std::vector<float> layerResults(layer.Size, 0.0f);

        for (int i = 0; i < layer.PrevLayerSize; i++) {
            for (int j = 0; j < layer.Size; j++) {
                layerResults[j] += (prevLayerActivations[i] * layer.WeightAt(i, j));
            }
        }

        LayerActivations layerData;
        layerData.LayerSize = layer.Size;
        layerData.PreActivations.resize(layer.Size);
        layerData.Activations.resize(layer.Size);

        for (int i = 0; i < layerResults.size(); i++) {
            layerData.PreActivations[i] = layerResults[i] + layer.Biases[i];
            layerResults[i] = m_model.m_activationFunc(layerResults[i] + layer.Biases[i]);
            layerData.Activations[i] = layerResults[i];
        }

        activations.Layers.push_back(std::move(layerData));
        prevLayerActivations = std::move(layerResults);
    }

    return activations;
}


ModelGradient ModelLearn::ComputeGradient(
    const ModelActivations& data, const Matrix<float>& layerErrors) const
{    
    const auto& modelLayers = m_model.m_layers;
    
    ModelGradient gradient;
    gradient.Layers.reserve(modelLayers.size());

    for (int layerIndex = 0; layerIndex < modelLayers.size(); layerIndex++) {
        ModelLayer layer(modelLayers[layerIndex].Size, modelLayers[layerIndex].PrevLayerSize);
        
        for (int i = 0; i < layer.Size; i++) {
            layer.Biases[i] = layerErrors[layerIndex][i];
        }

        const auto& prevLayer = data.Layers[layerIndex];
        for (int i = 0; i < layer.PrevLayerSize; i++) {
            for (int j = 0; j < layer.Size; j++) {
                layer.WeightAt(i, j) = prevLayer.Activations[i] * layerErrors[layerIndex][j];
            }
        }

        gradient.Layers.push_back(std::move(layer));
    }

    return gradient;
}


void ModelLearn::ApplyGradient(const ModelGradient& gradient)
{
    for (int layerIndex = 0; layerIndex < gradient.Layers.size(); layerIndex++) {
        auto& gradientLayer = gradient.Layers[layerIndex];
        auto& modelLayer = m_model.m_layers[layerIndex];

        for (int i = 0; i < modelLayer.Biases.size(); i++) {
            modelLayer.Biases[i] -= m_learnParams.LearnRate * gradientLayer.Biases[i];
        }

        for (int i = 0; i < modelLayer.Weights.size(); i++) {
            modelLayer.Weights[i] -= m_learnParams.LearnRate * gradientLayer.Weights[i];
        }
    }
}
