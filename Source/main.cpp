#include <cstdio>
 
#include "Pool.hpp"
#include "Model.hpp"
#include "ModelVisualizer.hpp"


int main() 
{
    Pool trainPool("Resources/Mnist/train-images", "Resources/Mnist/train-labels");
    Pool testPool("Resources/Mnist/test-images", "Resources/Mnist/test-labels");
    int numPixels = trainPool.GetElements()[0].Pixels.size();

    Model model;
    model.SetInputSize(numPixels);
    model.AddLayer(16);
    model.AddLayer(16);
    model.AddLayer(10);

    auto input = trainPool.GetElements()[0].Pixels;
    std::vector<float> res = model.Apply(input);
    for (float elem : res) {
        printf("%.2f ", elem);
    }
    printf("\n");

    printf("MSE loss: %.2f\n", model.GetMSELoss(trainPool));

    LayerToBmps(model.GetLayer(0), "Resources/Visualizations");

    return 0;
}