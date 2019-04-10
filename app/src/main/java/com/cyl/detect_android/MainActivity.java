package com.cyl.detect_android;

import android.graphics.Bitmap;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

//逻辑，先创建文件，之后把它封装成uri对象，之后封装带intent中，调用摄像头，之后调用过后会有一个
//返回结果，成功的话就把它显示出来
import android.content.Intent;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

//public class MainActivity extends AppCompatActivity {
//
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_main);
//    }
//}

public class MainActivity extends AppCompatActivity {
    public static final int TAKE_PHOTO=1;       //声明一个请求码，用于识别返回的结果
    private ImageView picture;
    private Uri imageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button takePhoto=findViewById(R.id.take_photo);         //takePhoto控件
        picture=findViewById(R.id.picture);
        takePhoto.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                File outputImage=new File(getExternalCacheDir(),"output_image.jpg");
                /*
                创建一个File文件对象，用于存放摄像头拍下的图片，我们把这个图片命名为output_image.jpg
                并把它存放在应用关联缓存目录下，调用getExternalCacheDir()可以得到这个目录，为什么要
                用关联缓存目录呢？由于android6.0开始，读写sd卡列为了危险权限，使用的时候必须要有权限，
                应用关联目录则可以跳过这一步
                 */
                try//判断图片是否存在，存在则删除在创建，不存在则直接创建
                {
                    if(outputImage.exists())
                    {
                        outputImage.delete();
                    }
                    outputImage.createNewFile();
                }
                catch (IOException e)
                {
                    e.printStackTrace();
                }
                if(Build.VERSION.SDK_INT >= 24)
                    //判断安卓的版本是否高于7.0，高于则调用高于的方法，低于则调用低于的方法
                    //把文件转换成Uri对象
                    /*
                    之所以这样，是因为android7.0以后直接使用本地真实路径是不安全的，会抛出异常。
                    FileProvider是一种特殊的内容提供器，可以对数据进行保护
                     */
                {
                    imageUri=FileProvider.getUriForFile(MainActivity.this,
                            "com.example.cameraalbumtest.fileprovider",outputImage);
                    /*
                    第一个参数：context对象
                    第二个参数：任意唯一的字符串
                    第三个参数：文件对象
                     */

                }
                else {
                    imageUri=Uri.fromFile(outputImage);
                }
                //使用隐示的Intent，系统会找到与它对应的活动，即调用摄像头，并把它存储
                //Intent intent=new intent("android.media.action.IMAGE_CAPTURE");
                Intent intent=new Intent("android.media.action.IMAGE_CAPTURE");
                intent.putExtra(MediaStore.EXTRA_OUTPUT,imageUri);
                startActivityForResult(intent,TAKE_PHOTO);
                //调用会返回结果的开启方式，返回成功的话，则把它显示出来
            }
        });
    }

    //处理返回结果的函数，下面是隐示Intent的返回结果的处理方式，具体见以前我所发的intent讲解
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        switch (requestCode)
        {
            case TAKE_PHOTO:
                if(resultCode==RESULT_OK)
                {
                    try
                    {
                        Bitmap bitmap= BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
                        picture.setImageBitmap(bitmap);
                        //将图片解析成Bitmap对象，并把它显现出来
                    }
                    catch (FileNotFoundException e)
                    {
                        e.printStackTrace();
                    }
                }
                break;
            default:
                break;
        }
    }
}
