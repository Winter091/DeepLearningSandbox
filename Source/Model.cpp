#include "Model.hpp"

#include <cstdlib>
#include <cstdio>
#include <random>
#include <algorithm>


Model::Model()
    : m_e2(std::random_device()())
    , m_random(-1, 1)
{}


void Model::SetInputSize(std::size_t size)
{
    m_inputSize = size;
}


void Model::AddLayer(std::size_t size)
{
    if (m_inputSize == 0) {
        fprintf(stderr, "Input size is not set\n");
        exit(EXIT_FAILURE);
    }

    std::size_t prevLayerSize = m_layers.empty() ? m_inputSize : m_layers.back().Size;
    AddLayerImpl(size, prevLayerSize);
}


void Model::AddLayerImpl(std::size_t size, std::size_t prevLayerSize)
{
    ModelLayer layer(size, prevLayerSize);

    for (std::size_t i = 0; i < prevLayerSize; i++) {        
        for (std::size_t j = 0; j < size; j++) {
            layer.WeightAt(i, j) = m_random(m_e2);
        }
    }

    for (std::size_t i = 0; i < size; i++) {
        layer.Biases[i] = 10.0f * m_random(m_e2);
    }

    m_layers.push_back(std::move(layer));
}


std::vector<float> Model::Apply(const std::vector<float>& input) const
{
    if (input.size() != m_inputSize) {
        fprintf(stderr, "Input size (%d) should be %d\n", input.size(), m_inputSize);
        exit(EXIT_FAILURE);
    }

    if (m_layers.empty()) {
        fprintf(stderr, "Modes has no layers");
        exit(EXIT_FAILURE);
    }

    auto sigmoidFunc = [](float x) -> float {
        return 1.0f / (1.0f + std::pow(M_E, -x));
    };

    auto transformFunc = [&sigmoidFunc](float x, float bias) -> float {
        return sigmoidFunc(x + bias);
    };
    
    std::vector<float> prevLayerResults = input;

    for (const auto& layer : m_layers) {
        std::vector<float> currResults(layer.Size, 0.0f);

        for (int i = 0; i < layer.PrevLayerSize; i++) {
            for (int j = 0; j < layer.Size; j++) {
                currResults[j] += (prevLayerResults[i] * layer.WeightAt(i, j));
            }
        }

        for (int i = 0; i < currResults.size(); i++) {
            currResults[i] = transformFunc(currResults[i], layer.Biases[i]);
        }

        prevLayerResults = std::move(currResults);
    }

    return prevLayerResults;
}


void Model::Fit(const Pool& learnPool, const Pool& testPool, const LearnParams& params)
{
    for (int i = 0; i < params.NumIters; i++) {
        DoLearnIteration(learnPool, testPool, params);
    }
}


void Model::DoLearnIteration(const Pool& learnPool, const Pool& testPool, const LearnParams& params)
{
    // Create batch to learn on
    // Apply model, remembering z^l and a^l
    // Compute error in the ouput layer
    // Compute error on previous layers
    // Having errors, compute gradient vector

    Pool batchLearnPool = learnPool.TakeRandom(params.BatchSize);

    std::vector<ModelLayer> batchGradient;

    for (const auto& record : batchLearnPool.GetElements()) {
        ModelLearnData data = ApplyForLearn(record.Features);
        data.target = record.Target;

        auto layerErrors = ComputeErrors(data);

        std::vector<ModelLayer> gradient = ComputeGradient(data, layerErrors);
        if (batchGradient.empty()) {
            batchGradient = std::move(gradient);
            continue;
        }
        
        for (int layerNum = 0; layerNum < gradient.size(); layerNum++) {
            auto& resLayer = batchGradient[layerNum];
            auto& thisLayer = gradient[layerNum];

            for (int i = 0; i < thisLayer.Biases.size(); i++) {
                resLayer.Biases[i] += thisLayer.Biases[i];
            }

            for (int i = 0; i < thisLayer.Weights.size(); i++) {
                resLayer.Weights[i] += thisLayer.Weights[i];
            }
        }
    }

    for (int layerNum = 0; layerNum < batchGradient.size(); layerNum++) {
        auto& layer = batchGradient[layerNum];

        for (int i = 0; i < layer.Biases.size(); i++) {
            layer.Biases[i] /= params.BatchSize;
        }

        for (int i = 0; i < layer.Weights.size(); i++) {
            layer.Weights[i] /= params.BatchSize;
        }
    }

    // batchIteration now contains gradient!

    // Apply antigradient
    for (int layerNum = 0; layerNum < batchGradient.size(); layerNum++) {
        auto& gradientLayer = batchGradient[layerNum];
        auto& modelLayer = m_layers[layerNum + 1];

        for (int i = 0; i < modelLayer.Biases.size(); i++) {
            modelLayer.Biases[i] -= gradientLayer.Biases[i];
        }

        for (int i = 0; i < modelLayer.Weights.size(); i++) {
            modelLayer.Weights[i] -= gradientLayer.Weights[i];
        }
    }
}


std::vector<std::vector<float>> Model::ComputeErrors(const ModelLearnData& data) const
{
    std::vector<float> resultLayerErrors = ComputeOutputLayerErrors(data);
    std::vector<std::vector<float>> layersErrors = ComputePrevLayersErrors(data, resultLayerErrors);

    return layersErrors;
}


std::vector<float> Model::ComputeOutputLayerErrors(const ModelLearnData& data) const
{
    std::vector<float> errors(m_layers.back().Size, 0.0f);

    std::vector<float> properSol(errors.size(), 0.0f);
    properSol[data.target] = 1.0f;

    auto sigmoidDerivative = [](float x) -> float {
        return std::pow(M_E, -x) / std::pow(1 + std::pow(M_E, -x), 2);
    };

    for (int i = 0; i < errors.size(); i++) {
        float a = data.LayerDatas.back().A[i];
        float y = properSol[i];
        float z = data.LayerDatas.back().Z[i];
        errors[i] = (2.0f * (a - y)) * sigmoidDerivative(z);
    }

    return errors;
}


std::vector<std::vector<float>> Model::ComputePrevLayersErrors(
    const ModelLearnData& data, 
    const std::vector<float>& resultLayerErrors) const 
{
    std::vector<std::vector<float>> result;
    result.reserve(data.LayerDatas.size());
    result.push_back(resultLayerErrors);

    auto sigmoidDerivative = [](float x) -> float {
        return std::pow(M_E, -x) / std::pow(1 + std::pow(M_E, -x), 2);
    };

    const std::vector<float>* nextLayerErrors = &resultLayerErrors;
    for (int i = data.LayerDatas.size() - 2; i >= 0; i--) {

        auto& nextLayer = m_layers[i + 1];
        std::vector<float> currLayerErrors(data.LayerDatas[i].LayerSize);

        for (int j = 0; j < currLayerErrors.size(); j++) {

            float sum = 0.0f;
            for (int k = 0; k < data.LayerDatas[i + 1].LayerSize; k++) {
                sum += (nextLayer.WeightAt(j, k) * nextLayerErrors->at(k));
            }

            float z = data.LayerDatas[i].Z[j];
            currLayerErrors[j] = sigmoidDerivative(z) * sum;

        }

        result.push_back(std::move(currLayerErrors));
        nextLayerErrors = &result.back();

    }

    std::vector<std::vector<float>> reversedResult;
    reversedResult.reserve(result.size());
    for (int i = 0; i <= result.size(); i++) {
        reversedResult[i] = std::move(result[result.size() - 1 - i]);
    }

    return reversedResult;
};


ModelLearnData Model::ApplyForLearn(const std::vector<float> input)
{
    if (input.size() != m_inputSize) {
        fprintf(stderr, "Input size (%d) should be %d\n", input.size(), m_inputSize);
        exit(EXIT_FAILURE);
    }

    if (m_layers.empty()) {
        fprintf(stderr, "Modes has no layers");
        exit(EXIT_FAILURE);
    }

    auto sigmoidFunc = [](float x) -> float {
        return 1.0f / (1.0f + std::pow(M_E, -x));
    };

    auto transformFunc = [&sigmoidFunc](float x, float bias) -> float {
        return sigmoidFunc(x + bias);
    };
    
    ModelLearnData data;
    // Input layer is not required
    data.LayerDatas.reserve(m_layers.size() - 1);

    std::vector<float> prevLayerResults = input;

    for (const auto& layer : m_layers) {
        std::vector<float> currResults(layer.Size, 0.0f);

        for (int i = 0; i < layer.PrevLayerSize; i++) {
            for (int j = 0; j < layer.Size; j++) {
                currResults[j] += (prevLayerResults[i] * layer.WeightAt(i, j));
            }
        }

        LayerLearnData layerData;
        layerData.LayerSize = layer.Size;
        layerData.Z.resize(layer.Size);
        layerData.A.resize(layer.Size);

        for (int i = 0; i < currResults.size(); i++) {
            layerData.Z[i] = currResults[i] + layer.Biases[i];
            currResults[i] = transformFunc(currResults[i], layer.Biases[i]);
            layerData.A[i] = currResults[i];
        }

        data.LayerDatas.push_back(std::move(layerData));
        prevLayerResults = std::move(currResults);
    }

    return data;
}


std::vector<ModelLayer> Model::ComputeGradient(const ModelLearnData& data, const std::vector<std::vector<float>>& layerErrors) const
{
    std::vector<ModelLayer> gradient;
    // Input layer is not required
    gradient.reserve(m_layers.size() - 1);

    for (int i = 1; i < m_layers.size(); i++) {
        ModelLayer layer(m_layers[i].Size, m_layers[i].PrevLayerSize);
        
        for (int j = 0; j < layer.Size; j++) {
            layer.Biases[j] = layerErrors[i][j];
        }

        const auto& prevLayer = data.LayerDatas[i - 1];
        for (int j = 0; j < layer.PrevLayerSize; j++) {
            for (int k = 0; k < layer.Size; k++) {
                layer.WeightAt(j, k) = prevLayer.A[j] * layerErrors[i][k];
            }
        }
    }

    return gradient;
}


const ModelLayer& Model::GetLayer(std::size_t index) const 
{ 
    if (index >= m_layers.size()) {
        fprintf(stderr, "Out of bounds layer get\n");
        exit(EXIT_FAILURE);
    }

    return m_layers[index];
}


float Model::GetMSELoss(const Pool& pool) const
{    
    auto lossFunc = [](std::vector<float> sol, std::vector<float> properSol) -> float {
        float sum = 0.0f;

        for (int i = 0; i < sol.size(); i++) {
            sum += std::pow(sol[i] - properSol[i], 2);
        }

        return sum;
    };
    
    float loss = 0.0f;

    for (const PoolElement& element : pool.GetElements()) {
        const auto& solution = Apply(element.Features);
        std::vector<float> properSolution(solution.size(), 0.0f);
        properSolution[element.Target] = 1.0f;

        loss += lossFunc(solution, properSolution);
    }

    return loss / pool.GetSize();
}
