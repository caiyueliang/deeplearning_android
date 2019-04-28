/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.env.Logger;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteFaceBoxesAPIModel implements Classifier {
  private static final String TAG = "TFObjectDetection";
  private static final Logger LOGGER = new Logger();

  // Only return this many results.
  private static final int NUM_DETECTIONS = 10;       // 返回的检测框最多个数
  // Float model
  private static final float IMAGE_MEAN = 128.0f;
  private static final float IMAGE_STD = 128.0f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;           // 线程个数
  private boolean isModelQuantized;                   // 模型是否量化
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;

  private ByteBuffer imgData;

  private Interpreter tfLite;
  private DataEncoder dataEncoder = new DataEncoder(1024);

  private TFLiteFaceBoxesAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    LOGGER.i("[loadModelFile] modelFilename " + modelFilename);
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static Classifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized)
      throws IOException {
    final TFLiteFaceBoxesAPIModel d = new TFLiteFaceBoxesAPIModel();

    InputStream labelsInput = null;
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    labelsInput = assetManager.open(actualFilename);
    BufferedReader br = null;
    br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();

    d.inputSize = inputSize;

    try {
      d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    d.tfLite.setNumThreads(NUM_THREADS);
    d.outputLocations = new float[1][NUM_DETECTIONS][4];
    d.outputClasses = new float[1][NUM_DETECTIONS][2];
    d.outputScores = new float[1][NUM_DETECTIONS];
    d.numDetections = new float[1];
    return d;
  }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        // 图片预处理
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = intValues[i * inputSize + j];
                if (isModelQuantized) {
                    // Quantized model  量化模型
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else {
                    // Float model      浮点模型
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    //imgData.putFloat(((pixelValue >> 16) & 0xFF));
                    //imgData.putFloat(((pixelValue >> 8) & 0xFF));
                    //imgData.putFloat((pixelValue & 0xFF));
                }
            }
        }
        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.   拷贝输入数据到TensorFlow
        Trace.beginSection("feed");
        // TODO
        outputLocations = new float[1][dataEncoder.getBoxesNum()][4];       // 输出坐标
        outputClasses = new float[1][dataEncoder.getBoxesNum()][2];         // 输出类别

        Object[] inputArray = {imgData};                      // 输入图片

        // TODO
        Map<Integer, Object> outputMap = new HashMap<>();     // 输出信息存放到outputMap，作为参数传给TF
        outputMap.put(0, outputClasses);
        outputMap.put(1, outputLocations);

        Trace.endSection();

        // Run the inference call.      模型推理
        Trace.beginSection("run");
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        float[][][] locPredict = (float[][][])outputMap.get(1);
        float[][][] classPredict = (float[][][])outputMap.get(0);
        Map<Integer, Object> output = dataEncoder.decode(locPredict[0], classPredict[0]);
        Trace.endSection();

        float[][] outputBoxes = (float[][]) output.get(0);
        float[] outputScores = (float[]) output.get(1);

        final ArrayList<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);

        int total = outputBoxes.length < NUM_DETECTIONS ? outputBoxes.length: NUM_DETECTIONS;
        for (int i = 0; i < total; ++i) {
            // (x1, y1, x2, y2) (left, top, right, bottom)
            final RectF detection =
                    new RectF(
                            outputBoxes[i][0] * inputSize,
                            outputBoxes[i][1] * inputSize,
                            outputBoxes[i][2] * inputSize,
                            outputBoxes[i][3] * inputSize);

            Log.i(TAG, String.format("filter box: l %f, t %f, r %f, b %f",
                    outputBoxes[i][0] * inputSize, outputBoxes[i][1] * inputSize,
                    outputBoxes[i][2] * inputSize, outputBoxes[i][3] * inputSize));

            // SSD Mobilenet V1 Model assumes class 0 is background class
            // in label file and class labels start from 1 to number_of_classes+1,
            // while outputClasses correspond to class index from 0 to number_of_classes
            int labelOffset = 1;
            recognitions.add(
                    new Recognition(
                            "" + i,
                            labels.get((int) outputClasses[0][i][0] + labelOffset),
                            outputScores[i],
                            detection));
        }
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    //    // TODO
    //    float[][][] locPredict = (float[][][])outputMap.get(1);
    //    for (int i = 0; i < 21824; i++) {
    //        Log.i(TAG, String.format("outputClasses %f, %f, %f, %f", locPredict[0][i][0],
    //                locPredict[0][i][1], locPredict[0][i][2], locPredict[0][i][3]));
    //    }
    //
    //    float[][][] classPredict = (float[][][])outputMap.get(0);
    //    for (int i = 0; i < 21824; i++) {
    //        Log.i(TAG, String.format("outputClasses %f, %f", classPredict[0][i][0], classPredict[0][i][1]));
    //    }

        // Show the best detections.
        // after scaling them back to the input size.
//        final ArrayList<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);
//        for (int i = 0; i < NUM_DETECTIONS; ++i) {
//            final RectF detection =
//                new RectF(
//                    outputLocations[0][i][1] * inputSize,
//                    outputLocations[0][i][0] * inputSize,
//                    outputLocations[0][i][3] * inputSize,
//                    outputLocations[0][i][2] * inputSize);
//            // SSD Mobilenet V1 Model assumes class 0 is background class
//            // in label file and class labels start from 1 to number_of_classes+1,
//            // while outputClasses correspond to class index from 0 to number_of_classes
//            int labelOffset = 1;
//            recognitions.add(
//                new Recognition(
//                    "" + i,
//                    labels.get((int) outputClasses[0][i][0] + labelOffset),
//                    outputScores[0][i],
//                    detection));
//        }
//        Trace.endSection(); // "recognizeImage"
//        return recognitions;
    }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {}

  public void setNumThreads(int num_threads) {
    if (tfLite != null) tfLite.setNumThreads(num_threads);
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) tfLite.setUseNNAPI(isChecked);
  }
}
