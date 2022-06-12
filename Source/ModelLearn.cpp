#include "ModelLearn.hpp"


ModelLearn::ModelLearn(Model& model)
    : m_model(model)
{}


void ModelLearn::Fit(const Pool& learnPool, const Pool& testPool, const LearnParams& params)
{
    for (int i = 0; i < params.NumIters; i++) {
        printf("Iteration %3d: mse loss = ", i);
        DoLearnIteration(learnPool, testPool, params);
        printf("%.3f\n", GetMSELoss(testPool));
    }
}


void ModelLearn::DoLearnIteration(
    const Pool& learnPool, const Pool& testPool, const LearnParams& params)
{
    // Create batch to learn on
    // Apply model, remembering z^l and a^l
    // Compute error in the ouput layer
    // Compute error on previous layers
    // Having errors, compute gradient vector

    Pool batchLearnPool = learnPool.TakeRandom(params.BatchSize);
    ModelGradient avgGradient;

    for (const auto& element : batchLearnPool.GetElements()) {
        ModelActivations activations = ComputeActivations(element.Features);

        Matrix<float> errors = ComputeErrors(activations, element.Target);

        ModelGradient gradient = ComputeGradient(activations, errors);
        if (avgGradient.Layers.empty()) {
            avgGradient = std::move(gradient);
            continue;
        }
        
        for (int layerNum = 0; layerNum < gradient.Layers.size(); layerNum++) {
            auto& resLayer = avgGradient.Layers[layerNum];
            auto& thisLayer = gradient.Layers[layerNum];

            for (int i = 0; i < thisLayer.Biases.size(); i++) {
                resLayer.Biases[i] += thisLayer.Biases[i];
            }

            for (int i = 0; i < thisLayer.Weights.size(); i++) {
                resLayer.Weights[i] += thisLayer.Weights[i];
            }
        }
    }

    for (int layerNum = 0; layerNum < avgGradient.Layers.size(); layerNum++) {
        auto& layer = avgGradient.Layers[layerNum];

        for (int i = 0; i < layer.Biases.size(); i++) {
            layer.Biases[i] /= params.BatchSize;
        }

        for (int i = 0; i < layer.Weights.size(); i++) {
            layer.Weights[i] /= params.BatchSize;
        }
    }

    // Apply antigradient
    for (int layerNum = 0; layerNum < avgGradient.Layers.size(); layerNum++) {
        auto& gradientLayer = avgGradient.Layers[layerNum];
        auto& modelLayer = m_model.m_layers[layerNum];

        for (int i = 0; i < modelLayer.Biases.size(); i++) {
            modelLayer.Biases[i] -= gradientLayer.Biases[i];
        }

        for (int i = 0; i < modelLayer.Weights.size(); i++) {
            modelLayer.Weights[i] -= gradientLayer.Weights[i];
        }
    }
}


Matrix<float> ModelLearn::ComputeErrors(
    const ModelActivations& activations, uint8_t target) const
{
    Matrix<float> result;
    result.reserve(m_model.m_layers.size());

    std::vector<float> outputLayerErrors = ComputeOutputLayerErrors(activations, target);
    result.push_back(std::move(outputLayerErrors));

    auto sigmoidDerivative = [](float x) -> float {
        return std::pow(M_E, -x) / std::pow(1 + std::pow(M_E, -x), 2);
    };

    const std::vector<float>* nextLayerErrors = &outputLayerErrors;
    for (int i = m_model.m_layers.size() - 2; i >= 0; i--) {

        auto& nextLayer = m_model.m_layers[i + 1];

        std::vector<float> currLayerErrors(m_model.m_layers[i].Size);

        for (int j = 0; j < currLayerErrors.size(); j++) {

            float sum = 0.0f;
            for (int k = 0; k < nextLayerErrors->size(); k++) {
                sum += (nextLayer.WeightAt(j, k) * nextLayerErrors->at(k));
            }

            // i + 1 because input layer is also stored in LayerDatas
            float z = activations.Layers[i + 1].PreActivations[j];
            currLayerErrors[j] = sigmoidDerivative(z) * sum;
        }

        result.push_back(std::move(currLayerErrors));
        nextLayerErrors = &result.back();

    }

    Matrix<float> reversedResult;
    reversedResult.reserve(result.size());
    for (int i = 0; i < result.size(); i++) {
        std::vector<float> a = std::move(result[result.size() - 1 - i]);
        reversedResult.push_back(std::move(a));
    }

    return reversedResult;
}


std::vector<float> ModelLearn::ComputeOutputLayerErrors(
    const ModelActivations& data, uint8_t target) const
{
    std::vector<float> errors(m_model.m_layers.back().Size, 0.0f);

    std::vector<float> correctSolution(errors.size(), 0.0f);
    correctSolution[target] = 1.0f;

    auto sigmoidDerivative = [](float x) -> float {
        return std::pow(M_E, -x) / std::pow(1 + std::pow(M_E, -x), 2);
    };

    for (int i = 0; i < errors.size(); i++) {
        float a = data.Layers.back().Activations[i];
        float y = correctSolution[i];
        float z = data.Layers.back().PreActivations[i];
        errors[i] = (2.0f * (a - y)) * sigmoidDerivative(z);
    }

    return errors;
}


ModelActivations ModelLearn::ComputeActivations(const std::vector<float> input)
{
    if (input.size() != m_model.m_inputSize) {
        fprintf(stderr, "Input size (%d) should be %d\n", input.size(), m_model.m_inputSize);
        exit(EXIT_FAILURE);
    }

    if (m_model.m_layers.empty()) {
        fprintf(stderr, "Modes has no layers");
        exit(EXIT_FAILURE);
    }

    auto sigmoidFunc = [](float x) -> float {
        return 1.0f / (1.0f + std::pow(M_E, -x));
    };

    auto transformFunc = [&sigmoidFunc](float x, float bias) -> float {
        return sigmoidFunc(x + bias);
    };
    
    ModelActivations data;
    data.Layers.reserve(m_model.m_layers.size() + 1);

    // Store a_i for input layer
    LayerActivations inputLayerData;
    inputLayerData.LayerSize = input.size();
    inputLayerData.Activations = input;
    data.Layers.push_back(std::move(inputLayerData));

    std::vector<float> prevLayerResults = input;

    for (const auto& layer : m_model.m_layers) {
        std::vector<float> currResults(layer.Size, 0.0f);

        for (int i = 0; i < layer.PrevLayerSize; i++) {
            for (int j = 0; j < layer.Size; j++) {
                currResults[j] += (prevLayerResults[i] * layer.WeightAt(i, j));
            }
        }

        LayerActivations layerData;
        layerData.LayerSize = layer.Size;
        layerData.PreActivations.resize(layer.Size);
        layerData.Activations.resize(layer.Size);

        for (int i = 0; i < currResults.size(); i++) {
            layerData.PreActivations[i] = currResults[i] + layer.Biases[i];
            currResults[i] = transformFunc(currResults[i], layer.Biases[i]);
            layerData.Activations[i] = currResults[i];
        }

        data.Layers.push_back(std::move(layerData));
        prevLayerResults = std::move(currResults);
    }

    return data;
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


float ModelLearn::GetMSELoss(const Pool& pool) const
{    
    auto lossFunc = [](const std::vector<float>& sol, const std::vector<float>& correctSol) {
        float sum = 0.0f;

        for (int i = 0; i < sol.size(); i++) {
            sum += std::pow(sol[i] - correctSol[i], 2);
        }

        return sum;
    };
    
    float loss = 0.0f;

    for (const PoolElement& element : pool.GetElements()) {
        const auto& solution = m_model.Apply(element.Features);
        std::vector<float> properSolution(solution.size(), 0.0f);
        properSolution[element.Target] = 1.0f;

        loss += lossFunc(solution, properSolution);
    }

    return loss / pool.GetSize();
}

