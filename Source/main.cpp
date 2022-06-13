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

    ModelVisualizer::DumpLayerToBmps(model.GetLayer(0), "Resources/Pre");

    auto input = testPool.GetElements()[0].Features;
    std::vector<float> res = model.Apply(input);
    for (float elem : res) {
        printf("%.2f ", elem);
    }
    printf("\n");

    model.Fit(trainPool, testPool, {
        .MaxIters = 1000000,
        .BatchSize = 32,
        .DesiredLoss = 0.39f,
        .LearnRate = 0.7f,
        .lossFunc = LossFunc::LogLoss,
        .activationFunc = ActivationFunc::Sigmoid,
    });

    input = testPool.GetElements()[0].Features;
    res = model.Apply(input);
    for (float elem : res) {
        printf("%.2f ", elem);
    }
    printf("\nTarget: %d\n", testPool.GetElements()[0].Target);

    ModelVisualizer::DumpLayerToBmps(model.GetLayer(0), "Resources/Post");

    return 0;
}