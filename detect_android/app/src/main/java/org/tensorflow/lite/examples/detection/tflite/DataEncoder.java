package org.tensorflow.lite.examples.detection.tflite;

import android.util.Log;

import java.util.ArrayList;

public class DataEncoder {
    private static final String TAG = "TFObjectDetection";
    private int boxesNum = 21824;
    private float scale;
    private float[] steps = new float[3];
    private float[] sizes = new float[3];
    private ArrayList<Integer[]> aspect_ratios = new ArrayList<Integer[]>();
    private int[] feature_map_sizes = new int[3];
    private ArrayList<Integer[]> density = new ArrayList<Integer[]>();
    private int num_layers = 3;
    private float[][] boxes = new float[this.boxesNum][4];     // [21824, 4] (cx, cy, w, h)

    public DataEncoder(float imageSize) {
        this.scale = imageSize;

        // steps = [s / scale for s in (32, 64, 128)]      // [0.03125, 0.0625, 0.125]
        this.steps[0] = 32 / this.scale;
        this.steps[1] = 64 / this.scale;
        this.steps[2] = 128 / this.scale;

        // sizes = [s / scale for s in (32, 256, 512)]     // [0.03125, 0.25, 0.5]     当32改为64时，achor与label匹配的正样本数目更多
        this.sizes[0] = 32 / this.scale;
        this.sizes[1] = 256 / this.scale;
        this.sizes[2] = 512 / this.scale;

        // aspect_ratios = ((1, 2, 4), (1,), (1,))
        this.aspect_ratios.add(new Integer[]{1, 2, 4});
        this.aspect_ratios.add(new Integer[]{1});
        this.aspect_ratios.add(new Integer[]{1});

        // feature_map_sizes = (32, 16, 8)
        this.feature_map_sizes[0] = 32;
        this.feature_map_sizes[0] = 16;
        this.feature_map_sizes[0] = 8;

        //density = [[-3, -1, 1, 3], [-1, 1], [0]]        // density for output layer1
        this.density.add(new Integer[]{-3, -1, 1, 3});
        this.density.add(new Integer[]{-1, 1});
        this.density.add(new Integer[]{0});

        int curBoxIndex = 0;

        for (int layerIndex = 0; layerIndex < 3; layerIndex++) {    // 遍历3层中的每一层
            int fmsize = this.feature_map_sizes[layerIndex];        // 分别为32, 16, 8

            // 生成32×32个，16×16个, 8×8个二元组，如：(0,0), (0,1), (0,2), ... (1,0), (1,1), ..., (32,32)
            for (float box_y = 0; box_y < fmsize; box_y++) {
                for (float box_x = 0; box_x < fmsize; box_x++) {
                    float center_x = (float)((box_x + 0.5) * this.steps[layerIndex]);   // 中心点坐标x
                    float center_y = (float)((box_y + 0.5) * this.steps[layerIndex]);   // 中心点坐标y

                    float s = sizes[layerIndex];
                    for (int ratios_i = 0; ratios_i < this.aspect_ratios.get(layerIndex).length; ratios_i++) {
                        Integer ar = this.aspect_ratios.get(layerIndex)[ratios_i];

                        if (layerIndex == 0) {
                            for (float dy = 0; dy < this.density.get(ratios_i).length; dy++) {
                                for (float dx = 0; dx < this.density.get(ratios_i).length; dx++) {
                                    this.boxes[curBoxIndex][0] = (float)(center_x + dx / 8.0 * s * ar);
                                    this.boxes[curBoxIndex][1] = (float)(center_y + dy / 8.0 * s * ar);
                                    this.boxes[curBoxIndex][2] = s * ar;
                                    this.boxes[curBoxIndex][3] = s * ar;
                                    curBoxIndex++;
                                }
                            }
                        } else  {
                            this.boxes[curBoxIndex][0] = center_x;
                            this.boxes[curBoxIndex][1] = center_y;
                            this.boxes[curBoxIndex][2] = s * ar;
                            this.boxes[curBoxIndex][3] = s * ar;
                            curBoxIndex++;
                        }
                    }
                }
            }
        }

        for (int i = 0; i < this.boxesNum; i++) {
            Log.i(TAG, String.format("%d: %f, %f, %f, %f", i, this.boxes[i][0], this.boxes[i][1],
                    this.boxes[i][2], this.boxes[i][3]));
        }
        Log.i(TAG, String.format("curBoxIndex %d", curBoxIndex));
    }

    public int getBoxesNum() {return this.boxesNum;}
}
