using System;
using System.Threading;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.TrackingModule;
using Rect = OpenCVForUnity.CoreModule.Rect;

using Utils = TensorFlowLite.Utils;

namespace TensorFlowLite
{
    public class HumanPose : BaseImagePredictor<float>
    {
        public enum Part
        {
            Head_top = 0, //0
            Thorax, //1
            R_Shoulder, //2
            R_Elbow, //3
            R_Wrist, // 4
            L_Shoulder, // 5
            L_Elbow, // 6
            L_Wrist, //7
            R_Hip, //8
            R_Knee, //9
            R_Ankle, //10
            L_Hip, // 11
            L_Knee, // 12
            L_Ankle, // 13
            Pelvis, //14
            Spine, //15
            Head, //16
            R_Hand, //17
            L_Hand, //18
            R_Toe, //19
            L_Toe //20
        }

        public static readonly Part[,] Connections = new Part[,]
        {
            // HEAD
            { Part.Head_top, Part.Head },
            { Part.Head, Part.Thorax },
            { Part.Thorax, Part.Spine },
            { Part.Spine, Part.Pelvis },
            // BODY
            { Part.Pelvis, Part.R_Hip },
            { Part.Pelvis, Part.L_Hip },
            { Part.R_Hip, Part.R_Knee },
            { Part.R_Knee, Part.R_Ankle },
            { Part.R_Ankle, Part.R_Toe },
            { Part.L_Hip, Part.L_Knee },
            { Part.L_Knee, Part.L_Ankle },
            { Part.L_Ankle, Part.L_Toe },
            { Part.Thorax, Part.R_Shoulder },
            { Part.R_Shoulder, Part.R_Elbow },
            { Part.R_Elbow, Part.R_Wrist },
            { Part.R_Wrist, Part.R_Hand },
            { Part.Thorax, Part.L_Shoulder },
            { Part.L_Shoulder, Part.L_Elbow },
            { Part.L_Elbow, Part.L_Wrist },
            { Part.L_Wrist, Part.L_Hand }
        };

        float[] focal = new float[]{ 1500, 1500 };
        float[] princpt = null;

        float root_depth = 12500;  // obtain this from RootNet (https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/tree/master/demo)

        float[,] img2bb_trans = new float[2, 3];
        float[,,] inputs0 = new float[3, Config.input_shape.Item1, Config.input_shape.Item2];
        float[,,] outputs0 = null;

        float[,] output_pose_2d = null;  // [21, 2]
        float[,] output_pose_3d = null;  // [21, 3]

        List<Scalar> colors = null;

        public HumanPose(string modelPath) : base(modelPath, Accelerator.NNAPI)
        {
            var odim0 = interpreter.GetOutputTensorInfo(0).shape;
            outputs0 = new float[odim0[1], odim0[2], odim0[3]];
            colors = Utils.GenerateColors(Config.joint_num);
        }

        public override void Invoke(Texture inputTex) {
            return;
        }

        public void Invoke(Mat inputMat, Rect bbox)
        {
            Preprocess(inputMat, bbox);
            
            interpreter.SetInputTensorData(0, inputs0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, outputs0);
            
            var pose_3d = Postprocess();
            output_pose_2d = pose_3d.Item1;
            output_pose_3d = pose_3d.Item2;
        }

        public void Preprocess(Mat inputMat, Rect bbox)
        {
            if (princpt == null) {
                princpt = new float[]{inputMat.width() / 2, inputMat.height() / 2};
            }
            
            Rect2d bbox2d = new Rect2d(bbox.x, bbox.y, bbox.width, bbox.height);
            Rect2d processed_bbox = Utils.ProcessBbox(bbox2d, inputMat.height(), inputMat.width(), Config.input_shape);
            var patch_image = Utils.GeneratePatchImage(inputMat, processed_bbox, false, 1.0f, 0.0f, false, Config.input_shape);

            Mat image = patch_image.Item1;
            Mat img2bb_trans_mat = patch_image.Item2;
            
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    img2bb_trans[i, j] = (float)img2bb_trans_mat.get(i, j)[0];
                }
            }

            image.convertTo(image, CvType.CV_32F, 1.0, 0);
            image = image - Config.pixel_mean;
            image = Utils.DivideByScalar(image, Config.pixel_std);
            Utils.MatToFloatArray(image, ref inputs0);
        }

        public (float[,], float[,]) Postprocess()
        {
            float[,] pose_3d = Utils.SoftArgmax(outputs0, Config.joint_num, Config.depth_dim, Config.output_shape);

            // Normalize the x and y coordinates
            for (int i = 0; i < Config.joint_num; i++)
            {
                pose_3d[i, 0] = pose_3d[i, 0] / Config.output_shape.Item2 * Config.input_shape.Item2;
                pose_3d[i, 1] = pose_3d[i, 1] / Config.output_shape.Item1 * Config.input_shape.Item1;
            }

            // Concatenate pose_3d with ones
            float[,] pose_3d_xy1 = new float[Config.joint_num, 3];
            for (int i = 0; i < Config.joint_num; i++)
            {
                pose_3d_xy1[i, 0] = pose_3d[i, 0];
                pose_3d_xy1[i, 1] = pose_3d[i, 1];
                pose_3d_xy1[i, 2] = 1;
            }

            // Concatenate img2bb_trans with [0, 0, 1]
            float[,] img2bb_trans_001 = new float[3, 3];
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    img2bb_trans_001[i, j] = img2bb_trans[i, j];
                }
            }
            img2bb_trans_001[2, 0] = 0;
            img2bb_trans_001[2, 1] = 0;
            img2bb_trans_001[2, 2] = 1;

            // Inverse img2bb_trans_001
            float[,] img2bb_trans_001_inv = Utils.InverseMatrix3x3(img2bb_trans_001);

            // Transpose img2bb_trans_001_inv
            for (int i = 0; i < 3; i++)
            {
                for (int j = i + 1; j < 3; j++)
                {
                    float temp = img2bb_trans_001_inv[i, j];
                    img2bb_trans_001_inv[i, j] = img2bb_trans_001_inv[j, i];
                    img2bb_trans_001_inv[j, i] = temp;
                }
            }

            // Multiply matrices
            float[,] pose_3d_transformed = Utils.MultiplyMatrix(pose_3d_xy1, img2bb_trans_001_inv);

            // Update the x and y coordinates
            for (int i = 0; i < Config.joint_num; i++)
            {
                pose_3d[i, 0] = pose_3d_transformed[i, 0];
                pose_3d[i, 1] = pose_3d_transformed[i, 1];
            }

            float[,] output_pose_2d = (float[,])pose_3d.Clone();

            // Calculate the absolute continuous depth
            for (int i = 0; i < Config.joint_num; i++)
            {
                pose_3d[i, 2] = (pose_3d[i, 2] / Config.depth_dim * 2 - 1) * (Config.bbox_3d_shape[0] / 2) + root_depth;
            }

            float[,] output_pose_3d = Utils.PixelToCam(pose_3d, focal, princpt);

            return (output_pose_2d, output_pose_3d);
        }

        public Mat DrawResult(Mat original_img) {
            return Utils.DrawResult(original_img, output_pose_2d, HumanPose.Connections, colors);
        }
    }
}
