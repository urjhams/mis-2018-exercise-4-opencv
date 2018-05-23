package com.example.mis.opencv;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";
    private CameraBridgeViewBase mOpenCvCameraView;
    Mat mRgba;      //here the matrix
    Mat imgGray;
    CascadeClassifier noseCascade,faceCascade;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    noseCascade = new CascadeClassifier(initAssetFile(getString(R.string.nose_cascade)));
                    faceCascade = new CascadeClassifier(initAssetFile(getString(R.string.face_cascade)));
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");

        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = findViewById(R.id.tutorial1_activity_java_surface_view);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null) mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG,
                    "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.d(TAG,
                    "Internal OpenCV library not found. " +
                            "Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0,
                    this,
                    mLoaderCallback);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        /* 8UC4: unsigned int 8bit (0 - 255), 4 color channels*/
        mRgba = new Mat(height,width, CvType.CV_8UC4);
        imgGray = new Mat(height,width,CvType.CV_8UC1);

    }

    public void onCameraViewStopped() {
        mRgba.release();
        imgGray.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        imgGray = inputFrame.gray();
        Imgproc.cvtColor(mRgba, imgGray, Imgproc.COLOR_RGB2GRAY);
        MatOfRect faces = new MatOfRect();
        if (faceCascade == null) {
            return mRgba;
        }
        if (noseCascade == null) {
            return mRgba;
        }
        faceCascade.detectMultiScale(imgGray,faces);
        for (Rect face : faces.toArray()) {
            Mat roi_gray = imgGray.submat(face);
            Mat roi_color = mRgba.submat(face);
            MatOfRect noses = new MatOfRect();
            noseCascade.detectMultiScale(roi_gray,noses);
            for (Rect nose : noses.toArray()) {
                Imgproc.circle(roi_color,
                    new Point(nose.x + nose.width/2 ,nose.y + nose.height/2),
                    (int) Math.ceil(nose.width/3),
                    new Scalar(255,0,0),
                    -1);
            }
        }
        return mRgba;
    }


    public String initAssetFile(String filename)  {
        File file = new File(getFilesDir(), filename);
        if (!file.exists()) try {
            InputStream is = getAssets().open(filename);
            OutputStream os = new FileOutputStream(file);
            byte[] data = new byte[is.available()];
            int result = is.read(data);
            Log.d(TAG,"fileReading result: "+result);
            os.write(data);
            is.close();
            os.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        Log.d(TAG,"prepared local file: "+filename);
        return file.getAbsolutePath();
    }
}
