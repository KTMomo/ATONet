import matplotlib.pyplot as plot
import numpy as np

data = [['SegNet', 16.7, 57],
        ['ENet', 135.4, 57],
        ['SQ', 16.7, 59.8],
        ['CRF-RNN', 1.4, 62.5],
        ['FCN-8s', 2.0, 63.1],
        ['FRNN', 2.1, 71.8],
        ['ERFNet', 41.7, 69.7],
        ['ICNet', 30.3, 70.6],
        ['TwoColumn', 14.7, 72.9],
        ['SwiftNetRN', 18.4, 76.5],
        ['LEDNet', 40, 70.6],
        ['BiSeNetV1-1', 105.8, 68.4],
        ['BiSeNetV1-2', 65.5, 74.7],
        ['DFANet A', 100, 71.3],
        ['DFANet B', 120, 67.1],
        ['BiSeNetV2', 156, 72.6],
        ['BiSeNetV2-L', 47.3, 75.3]]

plot.ylabel('Mean IoU (%)')
plot.xlabel('Inference Speed (FPS)')
plot.xticks(np.array([0, 50, 100, 150, 200]))
plot.yticks(np.array([0, 60, 65, 70, 75, 80]))

for item in data:
    plot.scatter(item[1], item[2], 20, 'black', 's')
    plot.text(item[1], item[2], item[0])

plot.grid()
plot.show()
