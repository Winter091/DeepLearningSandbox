#include "Model.hpp"

#include <cstdlib>
#include <cstdio>
#include <random>
#include <algorithm>

#include "ModelVisualizer.hpp"
#include "ModelLearn.hpp"


Model::Model()
    : m_e2(std::random_device()())
    , m_random(-1, 1)
{
    SetLossFunc(LossFunc::LogLoss);
    SetActivationFunc(ActivationFunc::Sigmoid);
}


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
        layer.Biases[i] = m_random(m_e2);
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
    
    std::vector<float> prevLayerResults = input;

    for (const auto& layer : m_layers) {
        std::vector<float> currResults(layer.Size, 0.0f);

        for (int i = 0; i < layer.PrevLayerSize; i++) {
            for (int j = 0; j < layer.Size; j++) {
                currResults[j] += (prevLayerResults[i] * layer.WeightAt(i, j));
            }
        }

        for (int i = 0; i < currResults.size(); i++) {
            currResults[i] = m_activationFunc(currResults[i] + layer.Biases[i]);
        }

        prevLayerResults = std::move(currResults);
    }

    return prevLayerResults;
}


void Model::Fit(const Pool& learnPool, const Pool& testPool, const LearnParams& params)
{
    ModelLearn learn(*this);
    learn.Fit(learnPool, testPool, params);
}


float Model::GetPoolLoss(const Pool& pool) const
{    
    float loss = 0.0f;

    for (const PoolElement& element : pool.GetElements()) {
        const auto& solution = Apply(element.Features);
        std::vector<float> properSolution(solution.size(), 0.0f);
        properSolution[element.Target] = 1.0f;

        loss += m_lossFunc(solution, properSolution);
    }

    return loss / pool.GetSize();
}


const ModelLayer& Model::GetLayer(std::size_t index) const 
{ 
    if (index >= m_layers.size()) {
        fprintf(stderr, "Out of bounds layer get\n");
        exit(EXIT_FAILURE);
    }

    return m_layers[index];
}


void Model::SetLossFunc(LossFunc lossFunc)
{
    switch (lossFunc) {
        case LossFunc::SquaredError:
        {
            m_lossFunc = [](const std::vector<float>& sol, const std::vector<float>& correctSol) {
                float sum = 0.0f;

                for (int i = 0; i < sol.size(); i++) {
                    sum += std::pow(sol[i] - correctSol[i], 2);
                }

                return sum;
            };

            m_lossFuncDerivative = [](float a, float y) -> float {
                return 2.0f * (a - y);
            };

            break;
        }

        case LossFunc::LogLoss:
        {
            m_lossFunc = [](const std::vector<float>& sol, const std::vector<float>& correctSol) {
                float sum = 0.0f;

                for (int i = 0; i < sol.size(); i++) {
                    sum += (correctSol[i]          * std::log(sol[i])) +
                           ((1.0f - correctSol[i]) * std::log(1.0f - sol[i]));
                }

                return -sum;
            };

            m_lossFuncDerivative = [](float a, float y) -> float {
                return (a - y) / (a * (1.0f - a));
            };

            break;
        }

        default:
        {
            fprintf(stderr, "Unknown loss function: %u", static_cast<uint32_t>(lossFunc));
            exit(EXIT_FAILURE);
        }
    }
}


void Model::SetActivationFunc(ActivationFunc activationFunc)
{
    switch (activationFunc) {
        case ActivationFunc::Sigmoid:
        {
            m_activationFunc = [](float x) -> float {
                return 1.0f / (1.0f + std::pow(M_E, -x));
            };

            m_activationFuncDerivative = [](float x) -> float {
                return std::pow(M_E, -x) / std::pow(1 + std::pow(M_E, -x), 2);
            };

            break;
        }

        default:
        {
            fprintf(stderr, "Unknown activation function: %u", static_cast<uint32_t>(activationFunc));
            exit(EXIT_FAILURE);
        }
    }
}