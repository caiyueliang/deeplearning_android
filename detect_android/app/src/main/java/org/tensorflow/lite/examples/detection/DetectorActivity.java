/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteFaceBoxesAPIModel;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();
    //private static final String MODEL_USE = "SSD";
    private static final String MODEL_USE = "FaceBoxes";

    // Configuration values for the prepackaged SSD model.
    private static int TF_OD_API_INPUT_SIZE = 300;
    private static boolean TF_OD_API_IS_QUANTIZED = true;
    private static String TF_OD_API_MODEL_FILE = "detect.tflite";
    private static String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    private static float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection.
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private Classifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private byte[] luminanceCopy;

    private BorderedText borderedText;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
            TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            // TODO
            if (MODEL_USE == "SSD") {
                TF_OD_API_INPUT_SIZE = 300;
                TF_OD_API_IS_QUANTIZED = true;
                TF_OD_API_MODEL_FILE = "detect.tflite";
                TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
                //TF_OD_API_MODEL_FILE = "face_ssd.tflite";
                //TF_OD_API_LABELS_FILE = "file:///android_asset/face_ssd_label.txt";
                MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;

                detector = TFLiteObjectDetectionAPIModel.create(
                                getAssets(),
                                TF_OD_API_MODEL_FILE,
                                TF_OD_API_LABELS_FILE,
                                TF_OD_API_INPUT_SIZE,
                                TF_OD_API_IS_QUANTIZED);
                cropSize = TF_OD_API_INPUT_SIZE;
            } else {
                TF_OD_API_INPUT_SIZE = 1024;
                TF_OD_API_IS_QUANTIZED = false;
                TF_OD_API_MODEL_FILE = "faceboxes_float.tflite";
                TF_OD_API_LABELS_FILE = "file:///android_asset/faceboxes_label.txt";
                MINIMUM_CONFIDENCE_TF_OD_API = 0.0f;

                detector = TFLiteFaceBoxesAPIModel.create(
                                getAssets(),
                                TF_OD_API_MODEL_FILE,
                                TF_OD_API_LABELS_FILE,
                                TF_OD_API_INPUT_SIZE,
                                TF_OD_API_IS_QUANTIZED);
                cropSize = TF_OD_API_INPUT_SIZE;
            }

        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e("Exception initializing classifier!", e);
            Toast toast =
                  Toast.makeText(
                      getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("[CYL] Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("[CYL] Initializing rgbFrameBitmap at size W x H: %d x %d", previewWidth, previewHeight);
        LOGGER.i("[CYL] Initializing croppedBitmap at size W x H: %d x %d", cropSize, cropSize);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);        // 存放原图？
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);                  // 存放变换后的图？（1024×1024）

        // 放回转换后的矩阵，(640, 480)转(1024, 1024)，并旋转一定角度(90度的倍数)
        // MAINTAIN_ASPECT为ture则不缩放大小，有必要会进行裁剪，这里为false
        frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                cropSize, cropSize,
                sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);      // 反转矩阵（cropToFrameTransform是逆操作，将(1024, 1024)转(640, 480)

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);    // 覆盖视图View
        trackingOverlay.addCallback(
            new DrawCallback() {
                @Override
                public void drawCallback(final Canvas canvas) {
                    tracker.draw(canvas);                                       // 画框的函数，追踪类里面，先追踪过滤，再画
                    if (isDebug()) {
                      tracker.drawDebug(canvas);
                    }
                }
            });
    }

  @Override
  protected void processImage() {
    ++timestamp;                                        // 时间戳累加
    final long currTimestamp = timestamp;               // 时间戳
    byte[] originalLuminance = getLuminance();          // 获取亮度

    // tracker第一次调用会初始化
    tracker.onFrame(
        previewWidth,
        previewHeight,
        getLuminanceStride(),
        sensorOrientation,
        originalLuminance,
        timestamp);
    trackingOverlay.postInvalidate();       // 本质是调用View的onDraw()绘制。主线程之外，用postInvalidate()。

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    // 将rgbFrameBitmap位图中的像素替换(填充)为getRgbBytes()返回的数组中的颜色值。数组中的每个元素都是int型（ARGB_8888格式）
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);  // 将rgbFrameBitmap画到view上
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    // 在后台线程里运行run()里面的内容
    runInBackground(
        new Runnable() {
          @Override
          public void run() {
              LOGGER.i("Running detection on image " + currTimestamp);
              final long startTime = SystemClock.uptimeMillis();

              // 执行图像识别
              final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
              lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

              cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
              final Canvas canvas = new Canvas(cropCopyBitmap);
              final Paint paint = new Paint();
              paint.setColor(Color.RED);
              paint.setStyle(Style.STROKE);
              paint.setStrokeWidth(2.0f);

              float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
              switch (MODE) {
                case TF_OD_API:
                  minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                  break;
              }

              final List<Classifier.Recognition> mappedRecognitions =
                  new LinkedList<Classifier.Recognition>();         // 存放识别结果?

              // 遍历每个结果
              for (final Classifier.Recognition result : results) {
                  final RectF location = result.getLocation();      // 获取坐标

                  // 判断坐标不为空并且置信度大于阈值
                  if (location != null && result.getConfidence() >= minimumConfidence) {
                      // canvas.drawRect(location, paint);             // 画检测框(无效)

                      // 将在(1024, 1024)的坐标值，转换（映射）成在(640, 480)上的坐标值，直接修改在location中
                      // LOGGER.i("[CYL] location old %s", location.toString());
                      cropToFrameTransform.mapRect(location);
                      // LOGGER.i("[CYL] location new %s", location.toString());

                      result.setLocation(location);
                      mappedRecognitions.add(result);               // 存放识别到的检测框，用于绘制
                  }
              }

              // 跟踪结果，过滤和画检测框
              tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
              trackingOverlay.postInvalidate();     // 本质是调用View的onDraw()绘制。主线程之外，用postInvalidate()。

              computingDetection = false;

              // UI界面上对应的控件上显示一些信息:坐标,耗时等
              runOnUiThread(
                  new Runnable() {
                      @Override
                      public void run() {
                          showFrameInfo(previewWidth + "x" + previewHeight);
                          showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                          showInference(lastProcessingTimeMs + "ms");
                      }
                  });
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}
