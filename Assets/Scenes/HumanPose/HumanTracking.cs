using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.TrackingModule;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityUtils.Helper;
using OpenCVForUnity.VideoModule;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using Rect = OpenCVForUnity.CoreModule.Rect;

namespace OpenCVForUnityExample
{
    /// <summary>
    /// Tracking Example
    /// An example of object tracking using the tracking (Tracking API) module.
    /// http://docs.opencv.org/trunk/d5/d07/tutorial_multitracker.html
    /// </summary>
    [RequireComponent(typeof(WebCamTextureToMatHelper))]
    public class HumanTracking : MonoBehaviour
    {
        /// <summary>
        /// The requested tracker dropdown.
        /// </summary>
        public Dropdown requestedTrackerDropdown;

        /// <summary>
        /// The requested resolution.
        /// </summary>
        public TrackerPreset requestedTracker = TrackerPreset._KCF;

        /// <summary>
        /// The requested resolution.
        /// </summary>
        public ResolutionPreset requestedResolution = ResolutionPreset._640x480;

        /// <summary>
        /// The requestedFPS.
        /// </summary>
        public FPSPreset requestedFPS = FPSPreset._30;

        /// <summary>
        /// The texture.
        /// </summary>
        Texture2D texture;

        /// <summary>
        /// The HOGDescriptor.
        /// </summary>
        HOGDescriptor des;

        /// <summary>
        /// The trackers.
        /// </summary>
        TrackerSetting tracker;

        /// <summary>
        /// The current frame index in the video stream % frameCheckTracker.
        /// </summary>
        int frameCurrent = 0;

        /// <summary>
        /// The frame index when the Intersection over Union (IoU) of the tracker and detector will be checked.
        /// </summary>
        int frameCheckTracker;

        /// <summary>
        /// The webcam texture to mat helper.
        /// </summary>
        WebCamTextureToMatHelper webCamTextureToMatHelper;

        /// <summary>
        /// The FPS monitor.
        /// </summary>
        FpsMonitor fpsMonitor;

        // Use this for initialization
        void Start()
        {
            fpsMonitor = GetComponent<FpsMonitor>();
            frameCheckTracker = (int)requestedFPS * 3;

            webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper>();
            int width, height;
            Dimensions(requestedResolution, out width, out height);
            webCamTextureToMatHelper.requestedWidth = width;
            webCamTextureToMatHelper.requestedHeight = height;
            webCamTextureToMatHelper.requestedFPS = (int)requestedFPS;
            webCamTextureToMatHelper.outputColorFormat = WebCamTextureToMatHelper.ColorFormat.RGB;
            webCamTextureToMatHelper.Initialize();

            // Update GUI state
            requestedTrackerDropdown.value = (int)requestedTracker;
        }

        /// <summary>
        /// Raises the video capture to mat helper initialized event.
        /// </summary>
        public void OnWebCamTextureToMatHelperInitialized()
        {
            Debug.Log("OnWebCamTextureToMatHelperInitialized");

            Mat webCamTextureMat = webCamTextureToMatHelper.GetMat();

            texture = new Texture2D(webCamTextureMat.cols(), webCamTextureMat.rows(), TextureFormat.RGB24, false);
            Utils.matToTexture2D(webCamTextureMat, texture);

            gameObject.GetComponent<Renderer>().material.mainTexture = texture;

            gameObject.transform.localScale = new Vector3(webCamTextureMat.cols(), webCamTextureMat.rows(), 1);
            Debug.Log("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);


            float width = webCamTextureMat.width();
            float height = webCamTextureMat.height();

            float widthScale = (float)Screen.width / width;
            float heightScale = (float)Screen.height / height;
            if (widthScale < heightScale)
            {
                Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
            }
            else
            {
                Camera.main.orthographicSize = height / 2;
            }

            des = new HOGDescriptor();
            des.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector());
            tracker = null;
        }

        /// <summary>
        /// Raises the video capture to mat helper disposed event.
        /// </summary>
        public void OnWebCamTextureToMatHelperDisposed()
        {
            Debug.Log("OnWebCamTextureToMatHelperDisposed");

            if (texture != null)
            {
                Texture2D.Destroy(texture);
                texture = null;
            }
        }

        /// <summary>
        /// Raises the webcam texture to mat helper error occurred event.
        /// </summary>
        /// <param name="errorCode">Error code.</param>
        public void OnWebCamTextureToMatHelperErrorOccurred(WebCamTextureToMatHelper.ErrorCode errorCode)
        {
            Debug.Log("OnWebCamTextureToMatHelperErrorOccurred " + errorCode);

            if (fpsMonitor != null)
            {
                fpsMonitor.consoleText = "ErrorCode: " + errorCode;
            }
        }

        // Update is called once per frame
        void Update()
        {
            if (!webCamTextureToMatHelper.IsInitialized())
                return;

            if (webCamTextureToMatHelper.IsPlaying() && webCamTextureToMatHelper.DidUpdateThisFrame())
            {
                Mat rgbMat = webCamTextureToMatHelper.GetMat();

                if (0 == frameCurrent || frameCurrent == frameCheckTracker || tracker == null) {

                    using (MatOfRect locations = new MatOfRect())
                    using (MatOfDouble weights = new MatOfDouble())
                    {
                        des.detectMultiScale(rgbMat, locations, weights);

                        if (locations.size().height == 0) {
                            return;
                        }

                        Rect region = locations.toArray()[0];

                        if (tracker == null || CalculateIoU(region, tracker.boundingBox) < 0.5) {
                            
                            ResetTrackers();

                            // init trackers.
                            switch (requestedTracker)
                            {
                                case TrackerPreset._KCF:
                                    TrackerKCF trackerKCF = TrackerKCF.create(new TrackerKCF_Params());
                                    trackerKCF.init(rgbMat, region);
                                    tracker = new TrackerSetting(trackerKCF, trackerKCF.GetType().Name.ToString(), new Scalar(255, 0, 0));
                                    break;
                                case TrackerPreset._CSRT:
                                    TrackerCSRT trackerCSRT = TrackerCSRT.create(new TrackerCSRT_Params());
                                    trackerCSRT.init(rgbMat, region);
                                    tracker = new TrackerSetting(trackerCSRT, trackerCSRT.GetType().Name.ToString(), new Scalar(0, 255, 0));
                                    break;
                                case TrackerPreset._MIL:
                                    TrackerMIL trackerMIL = TrackerMIL.create(new TrackerMIL_Params());
                                    trackerMIL.init(rgbMat, region);
                                    tracker = new TrackerSetting(trackerMIL, trackerMIL.GetType().Name.ToString(), new Scalar(0, 0, 255));
                                    break;
                            }
                        }
                    }
                }

                // update trackers.
                string label = tracker.label;
                Scalar lineColor = tracker.lineColor;
                Rect boundingBox = tracker.boundingBox;

                if (frameCurrent != 0) {
                    tracker.tracker.update(rgbMat, boundingBox);
                } 

                Imgproc.rectangle(rgbMat, boundingBox.tl(), boundingBox.br(), lineColor, 2, 1, 0);
                Imgproc.putText(rgbMat, label, new Point(boundingBox.x, boundingBox.y - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, lineColor, 1, Imgproc.LINE_AA, false);
                
                frameCurrent = ++frameCurrent % frameCheckTracker;
                Utils.matToTexture2D(rgbMat, texture);
            }
        }

        private void ResetTrackers()
        {
            if (tracker != null)
            {
                tracker.Dispose();
                tracker = null;
            }

            // requestedTrackerDropdown.interactable = true;
        }

        public float CalculateIoU(Rect box1, Rect box2)
        {
            // Determine the intersection rectangle
            float x_left = Mathf.Max(box1.x, box2.x);
            float y_top = Mathf.Max(box1.y, box2.y);
            float x_right = Mathf.Min(box1.x + box1.width, box2.x + box2.width);
            float y_bottom = Mathf.Min(box1.y + box1.height, box2.y + box2.height);

            // No intersection, IoU is zero
            if (x_right < x_left || y_bottom < y_top)
                return 0.0f;

            // Calculate the area of intersection rectangle
            float intersection_area = (x_right - x_left) * (y_bottom - y_top);

            // Calculate the area of both rectangles
            float box1_area = box1.width * box1.height;
            float box2_area = box2.width * box2.height;

            // Calculate the IoU
            float iou = intersection_area / (box1_area + box2_area - intersection_area);
            return iou;
        }

        /// <summary>
        /// Raises the requested resolution dropdown value changed event.
        /// </summary>
        public void OnRequestedTrackerDropdownValueChanged(int result)
        {
            if ((int)requestedTracker != result)
            {
                requestedTracker = (TrackerPreset)result;
            }

            frameCurrent = 0;
        }

        /// <summary>
        /// Raises the destroy event.
        /// </summary>
        void OnDestroy()
        {
            if (webCamTextureToMatHelper != null)
                webCamTextureToMatHelper.Dispose();

            if (des != null)
                des.Dispose();

            ResetTrackers();
        }

        /// <summary>
        /// Raises the back button click event.
        /// </summary>
        public void OnBackButtonClick()
        {
            SceneManager.LoadScene("OpenCVForUnityExample");
        }

        /// <summary>
        /// Raises the reset trackers button click event.
        /// </summary>
        public void OnResetTrackersButtonClick()
        {
            ResetTrackers();

            frameCurrent = 0;
        }

        /// <summary>
        /// Raises the change camera button click event.
        /// </summary>
        public void OnChangeCameraButtonClick()
        {
            webCamTextureToMatHelper.requestedIsFrontFacing = !webCamTextureToMatHelper.requestedIsFrontFacing;
        }

        class TrackerSetting
        {
            public Tracker tracker;
            public string label;
            public Scalar lineColor;
            public Rect boundingBox;

            public TrackerSetting(Tracker tracker, string label, Scalar lineColor)
            {
                this.tracker = tracker;
                this.label = label;
                this.lineColor = lineColor;
                this.boundingBox = new Rect();
            }

            public void Dispose()
            {
                if (tracker != null)
                {
                    tracker.Dispose();
                    tracker = null;
                }
            }
        }

        public enum TrackerPreset : int
        {
            _KCF = 0,
            _CSRT,
            _MIL,
        }

        public enum FPSPreset : int
        {
            _0 = 0,
            _1 = 1,
            _5 = 5,
            _10 = 10,
            _15 = 15,
            _30 = 30,
            _60 = 60,
        }

        public enum ResolutionPreset : byte
        {
            _50x50 = 0,
            _640x480,
            _1280x720,
            _1920x1080,
            _9999x9999,
        }

        private void Dimensions(ResolutionPreset preset, out int width, out int height)
        {
            switch (preset)
            {
                case ResolutionPreset._50x50:
                    width = 50;
                    height = 50;
                    break;
                case ResolutionPreset._640x480:
                    width = 640;
                    height = 480;
                    break;
                case ResolutionPreset._1280x720:
                    width = 1280;
                    height = 720;
                    break;
                case ResolutionPreset._1920x1080:
                    width = 1920;
                    height = 1080;
                    break;
                case ResolutionPreset._9999x9999:
                    width = 9999;
                    height = 9999;
                    break;
                default:
                    width = height = 0;
                    break;
            }
        }
    }
}