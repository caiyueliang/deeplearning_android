package org.tensorflow.lite.examples.detection.tflite;

import android.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class DataEncoder {
    private static final String TAG = "TFObjectDetection";
    private int boxesNum = 21824;
    private int layersNum = 3;

    private float scale;
    private float[] steps = new float[layersNum];
    private float[] sizes = new float[layersNum];
    private ArrayList<Integer[]> aspect_ratios = new ArrayList<Integer[]>();
    private int[] feature_map_sizes = new int[layersNum];
    private ArrayList<Integer[]> density = new ArrayList<Integer[]>();
    private float[][] boxes = new float[this.boxesNum][4];     // [21824, 4] (cx, cy, w, h)

    private NMS nms;

    public DataEncoder(float imageSize) {
        this.scale = imageSize;

        // steps = [s / scale for s in (32, 64, 128)]      // [0.03125, 0.0625, 0.125]
        this.steps[0] = (float)(32.0 / this.scale);
        this.steps[1] = (float)(64.0 / this.scale);
        this.steps[2] = (float)(128.0 / this.scale);

        // sizes = [s / scale for s in (32, 256, 512)]     // [0.03125, 0.25, 0.5]     当32改为64时，achor与label匹配的正样本数目更多
        this.sizes[0] = (float)(32.0 / this.scale);
        this.sizes[1] = (float)(256.0 / this.scale);
        this.sizes[2] = (float)(512.0 / this.scale);

        // aspect_ratios = ((1, 2, 4), (1,), (1,))
        this.aspect_ratios.add(new Integer[]{1, 2, 4});
        this.aspect_ratios.add(new Integer[]{1});
        this.aspect_ratios.add(new Integer[]{1});

        // feature_map_sizes = (32, 16, 8)
        this.feature_map_sizes[0] = 32;
        this.feature_map_sizes[1] = 16;
        this.feature_map_sizes[2] = 8;

        //density = [[-3, -1, 1, 3], [-1, 1], [0]]        // density for output layer1
        this.density.add(new Integer[]{-3, -1, 1, 3});
        this.density.add(new Integer[]{-1, 1});
        this.density.add(new Integer[]{0});

        int curBoxIndex = 0;

        for (int layerIndex = 0; layerIndex < this.layersNum; layerIndex++) {    // 遍历3层中的每一层
            int fmsize = this.feature_map_sizes[layerIndex];        // 分别为32, 16, 8

            // 生成32×32个，16×16个, 8×8个二元组，如：(0,0), (0,1), (0,2), ... (1,0), (1,1), ..., (32,32)
            for (float box_y = 0; box_y < fmsize; box_y++) {
                for (float box_x = 0; box_x < fmsize; box_x++) {
                    // cx = (w + 0.5)*steps[i]                     # 中心点坐标x
                    // cy = (h + 0.5)*steps[i]                     # 中心点坐标y
                    float center_x = (float)((box_x + 0.5) * this.steps[layerIndex]);   // 中心点坐标x
                    float center_y = (float)((box_y + 0.5) * this.steps[layerIndex]);   // 中心点坐标y

                    float s = this.sizes[layerIndex];
                    for (int ratios_i = 0; ratios_i < this.aspect_ratios.get(layerIndex).length; ratios_i++) {
                        Integer ar = this.aspect_ratios.get(layerIndex)[ratios_i];

                        if (layerIndex == 0) {
                            for (int dx_i = 0; dx_i < this.density.get(ratios_i).length; dx_i++) {
                                for (int dy_i = 0; dy_i < this.density.get(ratios_i).length; dy_i++) {

                                    float dx = this.density.get(ratios_i)[dx_i];
                                    float dy = this.density.get(ratios_i)[dy_i];

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

//        for (int i = 0; i < this.boxesNum; i++) {
//            Log.i(TAG, String.format("%d: %f, %f, %f, %f", i, this.boxes[i][0], this.boxes[i][1],
//                    this.boxes[i][2], this.boxes[i][3]));
//        }
//        Log.i(TAG, String.format("curBoxIndex %d", curBoxIndex));
//        Log.i(TAG, String.format("aspect_ratios %d", this.aspect_ratios.get(0).length));
//        Log.i(TAG, String.format("aspect_ratios %d", this.aspect_ratios.get(1).length));
//        Log.i(TAG, String.format("aspect_ratios %d", this.aspect_ratios.get(2).length));
//
//        Log.i(TAG, String.format("density %d", this.density.get(0).length));
//        Log.i(TAG, String.format("density %d", this.density.get(1).length));
//        Log.i(TAG, String.format("density %d", this.density.get(2).length));
    }

    public int getBoxesNum() {return this.boxesNum;}


//    public int[] nms(float[][] boxes, float[] scores, float threshold) {
//        float[] areas = new float[boxes.length];
//
//        for (int i = 0; i < boxes.length; i++) {
//            areas[i] = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]);
//        }
//
//        Arrays.sort(scores);
//        Log.i(TAG, String.format("curBoxIndex %d", curBoxIndex));
//
//    }
    // def nms(self, bboxes, scores, threshold=0.5):
    //     '''
    //     bboxes(tensor) [N,4]
    //     scores(tensor) [N,]
    //     '''
    //     x1 = bboxes[:,0]
    //     y1 = bboxes[:,1]
    //     x2 = bboxes[:,2]
    //     y2 = bboxes[:,3]
    //     areas = (x2-x1) * (y2-y1)
    //
    //     _,order = scores.sort(0,descending=True)
    //     keep = []
    //     while order.numel() > 0:
    //         i = order[0]
    //         keep.append(i)
    //
    //         if order.numel() == 1:
    //             break
    //
    //         xx1 = x1[order[1:]].clamp(min=x1[i].item())
    //         yy1 = y1[order[1:]].clamp(min=y1[i].item())
    //         xx2 = x2[order[1:]].clamp(max=x2[i].item())
    //         yy2 = y2[order[1:]].clamp(max=y2[i].item())
    //
    //         w = (xx2-xx1).clamp(min=0)
    //         h = (yy2-yy1).clamp(min=0)
    //         inter = w*h
    //
    //         ovr = inter / (areas[i] + areas[order[1:]] - inter)
    //         ids = (ovr<=threshold).nonzero().squeeze()
    //         if ids.numel() == 0:
    //             break
    //         order = order[ids+1]
    //     return torch.LongTensor(keep)



    // def decode(self, loc, conf, use_gpu, nms_threshold=0.5):
    //     '''
    //     將预测出的 loc/conf转换成真实的人脸框
    //     loc [21842, 4]
    //     conf [21824, 2]
    //     '''
    //     print('loc', loc.size(), loc)
    //     print('conf', conf.size(), conf)
    //     variances = [0.1, 0.2]
    //
    //     // variances = [0.1, 0.2]
    //     // cxcy = (boxes[:, :2] + boxes[:, 2:])/2 - default_boxes[:, :2]       // [21824,2]
    //     // cxcy /= variances[0] * default_boxes[:, 2:]
    //     // wh = (boxes[:, 2:] - boxes[:, :2]) / default_boxes[:, 2:]           // [21824,2]  为什么会出现0宽度？？
    //     // wh = torch.log(wh) / variances[1]                                   // Variable. log求自然对数
    //     if use_gpu:
    //         cxcy = loc[:, :2].cuda() * variances[0] * self.default_boxes[:, 2:].cuda() + self.default_boxes[:, :2].cuda()
    //         wh = torch.exp(loc[:, 2:] * variances[1]) * self.default_boxes[:, 2:].cuda()
    //         boxes = torch.cat([cxcy-wh/2, cxcy+wh/2], 1)                        // [21824,4]
    //     else:
    //         cxcy = loc[:, :2] * variances[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
    //         wh = torch.exp(loc[:, 2:] * variances[1]) * self.default_boxes[:, 2:]   // 返回一个新张量，包含输入input张量每个元素的指数
    //         boxes = torch.cat([cxcy-wh/2, cxcy+wh/2], 1)                        // [21824,4]
    //
    //     print('cxcy', cxcy.size(), cxcy)
    //     print('wh', wh.size(), wh)
    //     print('boxes', boxes.size(), boxes)
    //
    //     conf[:, 0] = 0.4    // 置信度第0列（背景）设为0.4，下面再取最大值，目的是为了过滤置信度小于0.4的标签
    //     max_conf, labels = conf.max(1)                          // [21842,1]
    //
    //     // print(max_conf)
    //     // print('labels', labels.long().sum().item())
    //     if labels.long().sum().item() is 0:                     // 标签和为0？表示图片没有标签？
    //         sconf, slabel = conf.max(0)
    //         max_conf[slabel[0:5]] = sconf[0:5]
    //         labels[slabel[0:5]] = 1
    //
    //     // print('labels', labels)
    //     ids = labels.nonzero().squeeze(1)
    //     print('ids', ids)
    //     // print('boxes', boxes.size(), boxes[ids])
    //
    //     // ids tensor([4301, 4303, 4322, 4324, 4972, 4974, 4993, 4995, 8126, 8147, 8168, 8777,
    //     //    8798, 8819, 8840, 8861, 9470, 9491, 9512])
    //     keep = self.nms(boxes[ids], max_conf[ids], nms_threshold)   // .squeeze(1))
    //     // keep tensor([10,  0])
    //
    //     return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep]
    public Map<Integer, Object> decode(float[][] loc, float[][] conf, float nms_threshold) {
        Map<Integer, Object> output = new HashMap<>();  // 返回输出结果

        float back_threshold = (float)0.4;              // 背景过滤置信度
        float[][] boxes = new float[][]

        for (int i = 0; i < loc.length; i++) {
            if (conf[i][0] > back_threshold) {

            }
        }

//        /*将list转化为Integer[]*/
//        ArrayList<Integer> intList = new ArrayList<Integer>();//泛型为Integer
//        intList.add(123);
//        intList.add(234);
//        intList.add(345);
//        Integer[] b = new Integer[intList.size()];//当泛型为Integer时，需要
//        b = (Integer[])intList.toArray(b);      //以Integer类来作为数组基本元素
//        System.out.println(Arrays.toString(b));//，否则输出会报错。
//
//        Integer[] c = (Integer[])intList.toArray();//此处编译没问题，但是涉及泛型
//        System.out.println(Arrays.toString(c)); //向下转换，运行会出错
//
//        System.out.println(c.getClass());
//        System.out.println(b.getClass());

        return output;
    }
}
