#include <cstdio>
 
#include "Pool.hpp"
#include "Model.hpp"
#include "ModelVisualizer.hpp"


int main() 
{
    Pool trainPool("Resources/Mnist/train-images", "Resources/Mnist/train-labels");
    Pool testPool("Resources/Mnist/test-images", "Resources/Mnist/test-labels");
    int numFeatures = trainPool.GetElements()[0].Features.size();

    Model model;
    model.SetInputSize(numFeatures);
    model.AddLayer(16);
    model.AddLayer(16);
    model.AddLayer(10);

    ModelVisualizer::DumpLayerToBmps(model.GetLayer(0), "Resources/Visualizations");

    auto input = testPool.GetElements()[0].Features;
    std::vector<float> res = model.Apply(input);
    for (float elem : res) {
        printf("%.2f ", elem);
    }
    printf("\n");

    LearnParams p;
    p.BatchSize = 100;
    p.NumIters = 100;
    model.Fit(trainPool, testPool, p);

    input = testPool.GetElements()[0].Features;
    res = model.Apply(input);
    for (float elem : res) {
        printf("%.2f ", elem);
    }
    printf("\nTarget: %d\n", testPool.GetElements()[0].Target);

    return 0;
}