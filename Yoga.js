import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';
import React, { useRef, useState, useEffect } from 'react'
import backend from '@tensorflow/tfjs-backend-webgl'
import Webcam from 'react-webcam'
import { count } from '../../utils/music';
import Instructions from '../../components/Instrctions/Instructions';
import { Link } from 'react-router-dom'

import './Yoga.css'


import DropDown from '../../components/DropDown/DropDown';
import { poseImages } from '../../utils/pose_images';
import { POINTS, keypointConnections } from '../../utils/data';
import { drawPoint, drawSegment } from '../../utils/helper'

const modelpath = 'http://localhost:3030/model.json';

let skeletonColor = 'rgb(255,255,255)'

let poseList = [
  'Chair','Cobra','Tree'
]


let interval

// flag variable is used to help capture the time when AI just detect 
// the pose as correct(probability more than threshold)
let flag = false


const modelPath = 'https://models.s3.jp-tok.cloud-object-storage.appdomain.cloud/model.json';

function Yoga() {
  const webcamRef = useRef(null)
  const canvasRef = useRef(null)


  const [startingTime, setStartingTime] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [poseTime, setPoseTime] = useState(0)
  const [bestPerform, setBestPerform] = useState(0)
  const [currentPose, setCurrentPose] = useState('Chair')
  const [isStartPose, setIsStartPose] = useState(false)
  const [errorMessages, setErrorMessages] = useState(['Start Pose'])
  const [currentView, setCurrentView] = useState('Front')

  const currentViewRef = useRef(currentView);
  const handleViewChange = (view) => {
    setCurrentView(view)
    currentViewRef.current = view
  }

  useEffect(() => {
    const timeDiff = (currentTime - startingTime)/1000
    if(flag) {
      setPoseTime(timeDiff)
    }
    if((currentTime - startingTime)/1000 > bestPerform) {
      setBestPerform(timeDiff)
    }
  }, [currentTime])


  useEffect(() => {
    setCurrentTime(0)
    setPoseTime(0)
    setBestPerform(0)
  }, [currentPose])

  const CLASS_NO = {
    Chair: 0,
    Cobra: 1,
    Tree: 6,
  }

  function get_center_point(landmarks, left_bodypart, right_bodypart) {
    let left = tf.gather(landmarks, left_bodypart, 1)
    let right = tf.gather(landmarks, right_bodypart, 1)
    const center = tf.add(tf.mul(left, 0.5), tf.mul(right, 0.5))
    return center
    
  }

  function get_pose_size(landmarks, torso_size_multiplier=2.5) {
    let hips_center = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP)
    let shoulders_center = get_center_point(landmarks,POINTS.LEFT_SHOULDER, POINTS.RIGHT_SHOULDER)
    let torso_size = tf.norm(tf.sub(shoulders_center, hips_center))
    let pose_center_new = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP)
    pose_center_new = tf.expandDims(pose_center_new, 1)

    pose_center_new = tf.broadcastTo(pose_center_new, [1, 17, 2])
    let d = tf.gather(tf.sub(landmarks, pose_center_new), 0, 0)
    let max_dist = tf.max(tf.norm(d,'euclidean', 0))
    let pose_size = tf.maximum(tf.mul(torso_size, torso_size_multiplier), max_dist)
    return pose_size
  }

  function normalize_pose_landmarks(landmarks) {
    let pose_center = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP)
    pose_center = tf.expandDims(pose_center, 1)
    pose_center = tf.broadcastTo(pose_center, 
        [1, 17, 2]
      )
    landmarks = tf.sub(landmarks, pose_center)

    let pose_size = get_pose_size(landmarks)
    landmarks = tf.div(landmarks, pose_size)
    return landmarks
  }

  function landmarks_to_embedding(landmarks) {
    landmarks = normalize_pose_landmarks(tf.expandDims(landmarks, 0))
    let embedding = tf.reshape(landmarks, [1,34])
    return embedding
  }

  const runMovenet = async () => {
    const detectorConfig = {modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER};
    const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
    const poseClassifier = await tf.loadLayersModel(modelPath)
    const countAudio = new Audio(count)
    countAudio.loop = true
    interval = setInterval(() => {
      detectPose(detector, poseClassifier, countAudio, currentViewRef.current)
    }, 100)
  }
  
  let utterance = null;
  let speechInProgress = false;

  const detectPose = async (detector, poseClassifier, countAudio, currentView) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      let notDetected = 0 
      const video = webcamRef.current.video
      const pose = await detector.estimatePoses(video)
      const ctx = canvasRef.current.getContext('2d')
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      try {
        const keypoints = pose[0].keypoints 
        // console.log(keypoints)
        let input = keypoints.map((keypoint) => {
          if(keypoint.score > 0.4) {
            if(!(keypoint.name === 'left_eye' || keypoint.name === 'right_eye')) {
              drawPoint(ctx, keypoint.x, keypoint.y, 8, 'rgb(255,255,255)')
              let connections = keypointConnections[keypoint.name]
              try {
                connections.forEach((connection) => {
                  let conName = connection.toUpperCase()
                  drawSegment(ctx, [keypoint.x, keypoint.y],
                      [keypoints[POINTS[conName]].x,
                       keypoints[POINTS[conName]].y]
                  , skeletonColor)
                })
              } catch(err) {

              }
              
            }
          } else {
            notDetected += 1
          } 
          return [keypoint.x, keypoint.y]
        })
        if(notDetected > 4) {
          skeletonColor = 'rgb(255,255,255)'
          return
        }

        const processedInput = landmarks_to_embedding(input)
        const classification = poseClassifier.predict(processedInput)
        function calculateAngle(point1, point2, point3) {
          const x1 = point1[0]; 
          const y1 = point1[1]; 
          const x2 = point2[0]; 
          const y2 = point2[1]; 
          const x3 = point3[0]; 
          const y3 = point3[1]; 
          const vector1 = [x2 - x1, y2 - y1]; 
          const vector2 = [x3 - x2, y3 - y2];
          const dotProduct = vector1[0] * vector2[0] + vector1[1] * vector2[1];
          const magnitude1 = Math.sqrt(vector1[0] ** 2 + vector1[1] ** 2);
          const magnitude2 = Math.sqrt(vector2[0] ** 2 + vector2[1] ** 2);
          const cosineAngle = dotProduct / (magnitude1 * magnitude2);
          const angleRadians = Math.acos(cosineAngle);
          const angleDegrees = (angleRadians * 180) / Math.PI;
          const positiveAngle = (angleDegrees + 360) % 360;
          return positiveAngle;
        }
        function calculateDistance(point1, point2) {
          const x1 = point1[0];
          const y1 = point1[1];
          const x2 = point2[0];
          const y2 = point2[1];
      
          const distance = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
          return distance;
        }
        function drawCircle(ctx, x, y, radius, color) {
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, 2 * Math.PI);
          ctx.strokeStyle = color;
          ctx.lineWidth = 5;
          ctx.stroke();
        }

        let errorMessages = []
        classification.array().then((data) => {         
          const classNo = CLASS_NO[currentPose]
          
          // console.log(input[POINTS["NOSE"]][0], input[POINTS["NOSE"]][1])
          if(data[0][classNo] > 0.97) {
            
            if(!flag) {
              // countAudio.play()
              setStartingTime(new Date(Date()).getTime())
              flag = true
            }
            setCurrentTime(new Date(Date()).getTime())
            errorMessages = []
            setErrorMessages(errorMessages)
            // ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height); 
            skeletonColor = 'rgb(0,255,0)'
          } else {
              if(classNo===6 && currentView==='Front') {
                if(calculateAngle(input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_ELBOW"]], input[POINTS["LEFT_WRIST"]]) > 45) {
                  skeletonColor = 'rgb(255,255,255)';
                  errorMessages.push ("Left arm angle is low");
                  drawCircle(ctx, keypoints[POINTS["LEFT_ELBOW"]].x, keypoints[POINTS["LEFT_ELBOW"]].y, 15, 'rgb(255,0,0)');
                }
                if(calculateAngle(input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_ELBOW"]], input[POINTS["RIGHT_WRIST"]]) > 45) {
                  skeletonColor = 'rgb(255,255,255)';
                  errorMessages.push("Right arm angle is low");
                  drawCircle(ctx, keypoints[POINTS["RIGHT_ELBOW"]].x, keypoints[POINTS["RIGHT_ELBOW"]].y, 15, 'rgb(255,0,0)');
                }
                if(calculateAngle(input[POINTS["RIGHT_ELBOW"]], input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_HIP"]]) > 45) {
                  skeletonColor = 'rgb(255,255,255)';
                  errorMessages.push("Rise Right Arm Higher");
                  drawCircle(ctx, keypoints[POINTS["RIGHT_SHOULDER"]].x, keypoints[POINTS["RIGHT_SHOULDER"]].y, 15, 'rgb(255,0,0)');
                }
                if(calculateAngle(input[POINTS["LEFT_ELBOW"]], input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_HIP"]]) > 45) {
                  skeletonColor = 'rgb(255,255,255)';
                  errorMessages.push("Rise Left Arm Higher");
                  drawCircle(ctx, keypoints[POINTS["LEFT_SHOULDER"]].x, keypoints[POINTS["LEFT_SHOULDER"]].y, 15, 'rgb(255,0,0)');
                }
                if(calculateDistance(input[POINTS["LEFT_WRIST"]],input[POINTS["RIGHT_WRIST"]])>30){
                  skeletonColor='rgb(255,255,255)';
                  errorMessages.push("Join your hands");
                  drawCircle(ctx, keypoints[POINTS["LEFT_WRIST"]].x, keypoints[POINTS["LEFT_WRIST"]].y, 15, 'rgb(255,0,0)');
                  drawCircle(ctx, keypoints[POINTS["RIGHT_WRIST"]].x, keypoints[POINTS["RIGHT_WRIST"]].y, 15, 'rgb(255,0,0)');
                }
                let rightLegAngle = calculateAngle(input[POINTS["RIGHT_HIP"]], input[POINTS["RIGHT_KNEE"]], input[POINTS["RIGHT_ANKLE"]]);
                let leftLegAngle = calculateAngle(input[POINTS["LEFT_HIP"]], input[POINTS["LEFT_KNEE"]], input[POINTS["LEFT_ANKLE"]]);
                if(rightLegAngle >=0 && rightLegAngle <= 10 && leftLegAngle >=0 && leftLegAngle <= 10) {
                  skeletonColor = 'rgb(255,255,255)';
                  errorMessages.push("Bend your knees");
                }
                else if(rightLegAngle<90 && leftLegAngle >=0 && leftLegAngle <= 10){
                  skeletonColor = 'rgb(255,255,255)';
                  errorMessages.push("Bend your Right knee");
                  drawCircle(ctx, keypoints[POINTS["RIGHT_KNEE"]].x, keypoints[POINTS["RIGHT_KNEE"]].y, 15, 'rgb(255,0,0)');
                }
                else if(leftLegAngle<90 && rightLegAngle >=0 && rightLegAngle <= 10){
                  skeletonColor = 'rgb(255,255,255)';
                  errorMessages.push("Bend your Left knee");
                  drawCircle(ctx, keypoints[POINTS["LEFT_KNEE"]].x, keypoints[POINTS["LEFT_KNEE"]].y, 15, 'rgb(255,0,0)');
                }
                setErrorMessages(errorMessages);
                
                if ('speechSynthesis' in window && errorMessages.length > 0) {
                  const synthesis = window.speechSynthesis;
                  utterance = new SpeechSynthesisUtterance(errorMessages.join('. '));
                  if (speechInProgress) {
                    window.speechSynthesis.cancel();
                    speechInProgress = false;
                  }
                  speechInProgress = true;
                  if(speechInProgress) {
                    synthesis.speak(utterance);
                  }
                } else {
                  console.log('Speech synthesis not supported in this browser.');
                }
              }
            
            if(classNo===6 && currentView==='Side') {
              if (calculateAngle(input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_ELBOW"]], input[POINTS["RIGHT_WRIST"]]) < 10 &&
                  calculateAngle(input[POINTS["LEFT_HIP"]], input[POINTS["LEFT_KNEE"]], input[POINTS["LEFT_ANKLE"]]) < 15 &&
                  calculateAngle(input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_HIP"]], input[POINTS["RIGHT_KNEE"]]) > 40 ||
                  calculateAngle(input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_ELBOW"]], input[POINTS["RIGHT_WRIST"]]) < 15 &&
                  calculateAngle(input[POINTS["LEFT_HIP"]], input[POINTS["LEFT_KNEE"]], input[POINTS["LEFT_ANKLE"]]) < 10 &&
                  calculateAngle(input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_HIP"]], input[POINTS["LEFT_KNEE"]]) > 40) {
                  skeletonColor = 'rgb(0,255,0)';
                }
                if(calculateAngle(input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_ELBOW"]], input[POINTS["RIGHT_WRIST"]]) > 10) {
                  skeletonColor = 'rgb(255,255,255)';
                  errorMessages.push("Straighten your arms");
                  drawCircle(ctx, keypoints[POINTS["RIGHT_ELBOW"]].x, keypoints[POINTS["RIGHT_ELBOW"]].y, 15, 'rgb(255,0,0)');
                }

                if(calculateAngle(input[POINTS["LEFT_HIP"]], input[POINTS["LEFT_KNEE"]], input[POINTS["LEFT_ANKLE"]]) > 15) {
                  skeletonColor = 'rgb(255,255,255)';
                  errorMessages.push("Straighten your leg");
                  drawCircle(ctx, keypoints[POINTS["LEFT_KNEE"]].x, keypoints[POINTS["LEFT_KNEE"]].y, 15, 'rgb(255,0,0)');
                }
                
                if(calculateAngle(input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_HIP"]], input[POINTS["RIGHT_KNEE"]]) < 40) {
                  skeletonColor = 'rgb(255,255,255)';
                  errorMessages.push("Bend your knee");
                  drawCircle(ctx, keypoints[POINTS["RIGHT_KNEE"]].x, keypoints[POINTS["RIGHT_KNEE"]].y, 15, 'rgb(255,0,0)');
                }
                
                setErrorMessages(errorMessages)
            }
            if(classNo===0 && currentView==='Side') {
              if(calculateAngle(input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_ELBOW"]], input[POINTS["RIGHT_WRIST"]]) > 45) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Straight your arms");
                drawCircle(ctx, keypoints[POINTS["RIGHT_ELBOW"]].x, keypoints[POINTS["RIGHT_ELBOW"]].y, 15, 'rgb(255,0,0)');
              }
              if(calculateAngle(input[POINTS["RIGHT_HIP"]], input[POINTS["RIGHT_KNEE"]], input[POINTS["RIGHT_ANKLE"]]) < 80) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Bend your knees angle");
                drawCircle(ctx, keypoints[POINTS["RIGHT_KNEE"]].x, keypoints[POINTS["RIGHT_KNEE"]].y, 15, 'rgb(255,0,0)');
              }
              if(calculateAngle(input[POINTS["RIGHT_HIP"]], input[POINTS["RIGHT_KNEE"]], input[POINTS["RIGHT_ANKLE"]]) > 95) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Rise your knee angle");
                drawCircle(ctx, keypoints[POINTS["RIGHT_KNEE"]].x, keypoints[POINTS["RIGHT_KNEE"]].y, 15, 'rgb(255,0,0)');
              }
              if(calculateAngle(input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_HIP"]], input[POINTS["RIGHT_KNEE"]]) > 100) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Lift your torso");
                drawCircle(ctx, keypoints[POINTS["RIGHT_HIP"]].x, keypoints[POINTS["RIGHT_HIP"]].y, 15, 'rgb(255,0,0)');
              }
              if(calculateAngle(input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_HIP"]], input[POINTS["RIGHT_KNEE"]]) < 70) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Bend your torso");
                drawCircle(ctx, keypoints[POINTS["RIGHT_HIP"]].x, keypoints[POINTS["RIGHT_HIP"]].y, 15, 'rgb(255,0,0)');
              }
              if(calculateAngle(input[POINTS["RIGHT_HIP"]], input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_ELBOW"]]) > 65) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Raise your arm angle");
                drawCircle(ctx, keypoints[POINTS["RIGHT_SHOULDER"]].x, keypoints[POINTS["RIGHT_SHOULDER"]].y, 15, 'rgb(255,0,0)');
              }
              setErrorMessages(errorMessages)


              // if ('speechSynthesis' in window && errorMessages.length > 0) {
              //     const synthesis = window.speechSynthesis
              //     utterance = new SpeechSynthesisUtterance(errorMessages.join('. '))
              //     if (speechInProgress) {
              //       window.speechSynthesis.cancel()
              //       speechInProgress = false
              //     }
              //     speechInProgress = true
              //     if(speechInProgress) {
              //       synthesis.speak(utterance)
              //     }
              //   } else {
              //     console.log('Speech synthesis not supported in this browser.')
              // }

            }

            if(classNo===0 && currentView==='Front') {
              if (calculateAngle(input[POINTS["RIGHT_ANKLE"]], input[POINTS["RIGHT_KNEE"]], input[POINTS["RIGHT_HIP"]]) > 10) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Bend your leg");
                drawCircle(ctx, keypoints[POINTS["RIGHT_KNEE"]].x, keypoints[POINTS["RIGHT_KNEE"]].y, 20, 'rgb(255,0,0)');
              }
              if (calculateAngle(input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_ELBOW"]], input[POINTS["LEFT_WRIST"]]) > 20) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Straighten your hand");
                drawCircle(ctx, keypoints[POINTS["RIGHT_ELBOW"]].x, keypoints[POINTS["RIGHT_ELBOW"]].y, 20, 'rgb(255,0,0)');
              }
              if (calculateAngle(input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_HIP"]], input[POINTS["LEFT_KNEE"]]) > 8) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Bend your HIP");
                drawCircle(ctx, keypoints[POINTS["RIGHT_ELBOW"]].x, keypoints[POINTS["RIGHT_ELBOW"]].y, 20, 'rgb(255,0,0)');
              }
              if(calculateDistance(input[POINTS["LEFT_KNEE"]],input[POINTS["RIGHT_KNEE"]]) > 35){
                skeletonColor='rgb(255,255,255)';
                errorMessages.push("Join your knees");
                drawCircle(ctx, keypoints[POINTS["LEFT_KNEE"]].x, keypoints[POINTS["LEFT_KNEE"]].y, 15, 'rgb(255,0,0)');
                drawCircle(ctx, keypoints[POINTS["RIGHT_KNEE"]].x, keypoints[POINTS["RIGHT_KNEE"]].y, 15, 'rgb(255,0,0)');
              }
              if(calculateDistance(input[POINTS["LEFT_ANKLE"]],input[POINTS["RIGHT_ANKLE"]]) > 20){
                skeletonColor='rgb(255,255,255)';
                errorMessages.push("Join your ankles");
                drawCircle(ctx, keypoints[POINTS["LEFT_ANKLE"]].x, keypoints[POINTS["LEFT_ANKLE"]].y, 15, 'rgb(255,0,0)');
                drawCircle(ctx, keypoints[POINTS["RIGHT_ANKLE"]].x, keypoints[POINTS["RIGHT_ANKLE"]].y, 15, 'rgb(255,0,0)');
              }
              if(calculateDistance(input[POINTS["LEFT_WRIST"]],input[POINTS["RIGHT_WRIST"]]) > 35){
                skeletonColor='rgb(255,255,255)';
                errorMessages.push("Join your wrists");
                drawCircle(ctx, keypoints[POINTS["LEFT_WRIST"]].x, keypoints[POINTS["LEFT_WRIST"]].y, 15, 'rgb(255,0,0)');
                drawCircle(ctx, keypoints[POINTS["RIGHT_WRIST"]].x, keypoints[POINTS["RIGHT_WRIST"]].y, 15, 'rgb(255,0,0)');
              }
              if(calculateAngle(input[POINTS["RIGHT_ANKLE"]], input[POINTS["RIGHT_KNEE"]], input[POINTS["RIGHT_HIP"]]) < 10 &&
              calculateAngle(input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_ELBOW"]], input[POINTS["LEFT_WRIST"]]) < 20 &&
              calculateAngle(input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_HIP"]], input[POINTS["LEFT_KNEE"]])  < 8 &&
              calculateDistance(input[POINTS["LEFT_KNEE"]],input[POINTS["RIGHT_KNEE"]]) < 35 &&
              calculateDistance(input[POINTS["LEFT_ANKLE"]],input[POINTS["RIGHT_ANKLE"]]) < 20 &&
              calculateDistance(input[POINTS["LEFT_WRIST"]],input[POINTS["RIGHT_WRIST"]]) < 35){
                skeletonColor = 'rgb(0,255,0)';
              }
              setErrorMessages(errorMessages)
            }

            if(classNo===1 && currentView==='Side') {
              if (calculateAngle(input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_ELBOW"]], input[POINTS["RIGHT_WRIST"]]) > 20) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Straighten right arm");
                drawCircle(ctx, keypoints[POINTS["RIGHT_ELBOW"]].x, keypoints[POINTS["RIGHT_ELBOW"]].y, 20, 'rgb(255,0,0)');
              }
              if (calculateAngle(input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_ELBOW"]], input[POINTS["LEFT_WRIST"]]) > 20) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Straight Left arm");
                drawCircle(ctx, keypoints[POINTS["LEFT_ELBOW"]].x, keypoints[POINTS["LEFT_ELBOW"]].y, 20, 'rgb(255,0,0)');
              }
              if (calculateAngle(input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_HIP"]], input[POINTS["RIGHT_KNEE"]]) < 30) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Straighten right hip");
                drawCircle(ctx, keypoints[POINTS["RIGHT_HIP"]].x, keypoints[POINTS["RIGHT_HIP"]].y, 20, 'rgb(255,0,0)');
              }
              if (calculateAngle(input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_HIP"]], input[POINTS["LEFT_KNEE"]]) < 30) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Straighten left gaand");
                drawCircle(ctx, keypoints[POINTS["LEFT_HIP"]].x, keypoints[POINTS["LEFT_HIP"]].y, 20, 'rgb(255,0,0)');
              }
              if (calculateAngle(input[POINTS["LEFT_ELBOW"]], input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_HIP"]]) < 130) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Straighten left arm");
                drawCircle(ctx, keypoints[POINTS["LEFT_SHOULDER"]].x, keypoints[POINTS["LEFT_SHOULDER"]].y, 20, 'rgb(255,0,0)');
              }
              if (calculateAngle(input[POINTS["RIGHT_ELBOW"]], input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_HIP"]]) < 130) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Make hands perpendicular");
                drawCircle(ctx, keypoints[POINTS["RIGHT_SHOULDER"]].x, keypoints[POINTS["RIGHT_SHOULDER"]].y, 20, 'rgb(255,0,0)');
              }
              if (calculateAngle(input[POINTS["RIGHT_ANKLE"]], input[POINTS["RIGHT_KNEE"]], input[POINTS["RIGHT_HIP"]]) > 30) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Straighten your legs");
                drawCircle(ctx, keypoints[POINTS["RIGHT_KNEE"]].x, keypoints[POINTS["RIGHT_KNEE"]].y, 20, 'rgb(255,0,0)');
              }
              if (calculateAngle(input[POINTS["LEFT_ANKLE"]], input[POINTS["LEFT_KNEE"]], input[POINTS["LEFT_HIP"]]) > 30) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Straigthen your  legs");
                drawCircle(ctx, keypoints[POINTS["LEFT_KNEE"]].x, keypoints[POINTS["LEFT_KNEE"]].y, 20, 'rgb(255,0,0)');
              }
              if (calculateAngle(input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_ELBOW"]], input[POINTS["RIGHT_WRIST"]]) < 20 &&
                calculateAngle(input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_ELBOW"]], input[POINTS["LEFT_WRIST"]]) < 20 &&
                calculateAngle(input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_HIP"]], input[POINTS["RIGHT_KNEE"]]) > 30 &&
                calculateAngle(input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_HIP"]], input[POINTS["LEFT_KNEE"]]) > 30 &&
                calculateAngle(input[POINTS["LEFT_ELBOW"]], input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_HIP"]]) > 130 &&
                calculateAngle(input[POINTS["RIGHT_ELBOW"]], input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_HIP"]]) > 130 &&
                calculateAngle(input[POINTS["RIGHT_ANKLE"]], input[POINTS["RIGHT_KNEE"]], input[POINTS["RIGHT_HIP"]]) < 30 &&
                calculateAngle(input[POINTS["LEFT_ANKLE"]], input[POINTS["LEFT_KNEE"]], input[POINTS["LEFT_HIP"]]) < 30) {
                skeletonColor = 'rgb(0,255,0)';
              }
              setErrorMessages(errorMessages)
            }

            if(classNo===1 && currentView==='Front') {
              if (calculateAngle(input[POINTS["RIGHT_WRIST"]], input[POINTS["RIGHT_ELBOW"]], input[POINTS["RIGHT_SHOULDER"]]) < 0 || 
              calculateAngle(input[POINTS["RIGHT_WRIST"]], input[POINTS["RIGHT_ELBOW"]], input[POINTS["RIGHT_SHOULDER"]])>8) {
              skeletonColor = 'rgb(255,255,255)';
              errorMessages.push("Straighten right arm");
              drawCircle(ctx, keypoints[POINTS["RIGHT_ELBOW"]].x, keypoints[POINTS["RIGHT_ELBOW"]].y, 20, 'rgb(255,0,0)');
            }
              if(calculateAngle(input[POINTS["LEFT_WRIST"]], input[POINTS["LEFT_ELBOW"]], input[POINTS["LEFT_SHOULDER"]]) <0 ||
              calculateAngle(input[POINTS["LEFT_WRIST"]], input[POINTS["LEFT_ELBOW"]], input[POINTS["LEFT_SHOULDER"]])>8) {
              skeletonColor = 'rgb(255,255,255)';
              errorMessages.push("Straighten left arm");
              drawCircle(ctx, keypoints[POINTS["LEFT_ELBOW"]].x, keypoints[POINTS["LEFT_ELBOW"]].y, 20, 'rgb(255,0,0)');
            }
            if (calculateDistance(input[POINTS["LEFT_WRIST"]], input[POINTS["RIGHT_WRIST"]]) > 250 ||
               200>calculateDistance(input[POINTS["LEFT_WRIST"]], input[POINTS["RIGHT_WRIST"]])){
              skeletonColor='rgb(255,255,255)';
              errorMessages.push("Adjust your hands");
              drawCircle(ctx, keypoints[POINTS["LEFT_WRIST"]].x, keypoints[POINTS["LEFT_WRIST"]].y, 20, 'rgb(255,0,0)');
              drawCircle(ctx, keypoints[POINTS["RIGHT_WRIST"]].x, keypoints[POINTS["RIGHT_WRIST"]].y, 20, 'rgb(255,0,0)');
            }
            if (calculateAngle(input[POINTS["RIGHT_HIP"]], input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_ELBOW"]])<162) {
              skeletonColor = 'rgb(255,255,255)';
              errorMessages.push("Lower your body");
              drawCircle(ctx, keypoints[POINTS["RIGHT_HIP"]].x, keypoints[POINTS["RIGHT_HIP"]].y, 20, 'rgb(255,0,0)');
            }
              if (calculateAngle(input[POINTS["LEFT_HIP"]], input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_ELBOW"]]) < 162) {
                skeletonColor = 'rgb(255,255,255)';
                errorMessages.push("Lower your body");
                drawCircle(ctx, keypoints[POINTS["LEFT_HIP"]].x, keypoints[POINTS["LEFT_HIP"]].y, 20, 'rgb(255,0,0)');
              }
              if (calculateAngle(input[POINTS["RIGHT_WRIST"]], input[POINTS["RIGHT_ELBOW"]], input[POINTS["RIGHT_SHOULDER"]]) > 0 &&
                calculateAngle(input[POINTS["RIGHT_WRIST"]], input[POINTS["RIGHT_ELBOW"]], input[POINTS["RIGHT_SHOULDER"]]) < 8 &&
                calculateAngle(input[POINTS["LEFT_WRIST"]], input[POINTS["LEFT_ELBOW"]], input[POINTS["LEFT_SHOULDER"]]) > 0 &&
                  calculateAngle(input[POINTS["LEFT_WRIST"]], input[POINTS["LEFT_ELBOW"]], input[POINTS["LEFT_SHOULDER"]]) < 8 &&
                calculateDistance(input[POINTS["LEFT_WRIST"]], input[POINTS["RIGHT_WRIST"]]) < 250 &&
                200 < calculateDistance(input[POINTS["LEFT_WRIST"]], input[POINTS["RIGHT_WRIST"]]) &&
                calculateAngle(input[POINTS["RIGHT_HIP"]], input[POINTS["RIGHT_SHOULDER"]], input[POINTS["RIGHT_ELBOW"]]) >162 &&
                calculateAngle(input[POINTS["LEFT_HIP"]], input[POINTS["LEFT_SHOULDER"]], input[POINTS["LEFT_ELBOW"]]) >162){
                  skeletonColor = 'rgb(0,255,0)';
                }
              setErrorMessages(errorMessages)
            }

            flag = false
            // skeletonColor = 'rgb(255,255,255)'
            // countAudio.pause()
            countAudio.currentTime = 0
          }
        })
      } catch(err) {
        console.log(err)
      }   
    }
  }

  function startYoga(){
    setIsStartPose(true) 
    runMovenet()
  } 

  function stopPose() {
    setIsStartPose(false)
    clearInterval(interval)
  }


  if(isStartPose) {
    return (
      <div className="yoga-container">
        <fieldset>
        <div class="toggle">
          <input 
            type="radio"
            value="Front"
            checked={currentView === 'Front'}
            onChange={() => handleViewChange('Front')}
            id="front"
          />
          <label for="front">Front View</label>
          <input 
            type="radio"
            value="Side"
            checked={currentView === 'Side'}
            onChange={() => handleViewChange('Side')} 
            id="side"
          />
          <label for="side">Side View</label>
        </div>
        </fieldset>
        
        <div>
          <Webcam 
          width='640px'
          height='480px'
          id="webcam"
          ref={webcamRef}
          style={{
            position: 'absolute',
            left: 120,
            top: 100,
            padding: '0px',
          }}
          />
          <canvas
            ref={canvasRef}
            id="my-canvas"
            width='640px'
            height='480px'
            style={{
              position: 'absolute',
              left: 120,
              top: 100,
              zIndex: 1
            }}
          >
          </canvas>
          <div>
              <img 
                src={poseImages[currentPose]}
                className="pose-img"
              />
          </div>
          <div className="error-info">
            {errorMessages.length > 0 && (
            <ul>
                {errorMessages.map((error, index) => (
                    <li key={index}>{error}</li>
                ))}
            </ul>
            )}
          </div>
        </div>
          <button
            onClick={stopPose}
            className="secondary-btn"    
          >Stop Pose</button>
      </div>
    )
  }

  return (
    <div
      className="yoga-container"
    >
      <Link to='/'>
                    <button 
                        className='btn'
                        id="home-btn"
                    >
                        Home
                    </button>
                </Link>
      <DropDown
        poseList={poseList}
        currentPose={currentPose}
        setCurrentPose={setCurrentPose}
      />
      <Instructions
          currentPose={currentPose}
        />
      <button
          onClick={startYoga}
          className="secondary-btn"    
        >Start Pose</button>
    </div>
  )
}

export default Yoga