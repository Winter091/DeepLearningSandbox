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
            layer.Weights[i][j] = m_random(m_e2);
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
                currResults[j] += (prevLayerResults[i] * layer.Weights[i][j]);
            }
        }

        for (int i = 0; i < currResults.size(); i++) {
            currResults[i] = transformFunc(currResults[i], layer.Biases[i]);
        }

        prevLayerResults = std::move(currResults);
    }

    return prevLayerResults;
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
