/* IMPORTS */
import qupath.lib.images.servers.TransformedServerBuilder
import qupath.lib.roi.interfaces.ROI
import qupath.imagej.tools.IJTools
import qupath.lib.images.PathImage
import qupath.lib.regions.RegionRequest
import ij.ImagePlus
import ij.process.ImageProcessor
import qupath.opencv.ml.pixel.PixelClassifiers
import qupath.lib.gui.viewer.OverlayOptions
import qupath.lib.gui.viewer.RegionFilter
import qupath.lib.gui.viewer.overlays.PixelClassificationOverlay
import qupath.lib.images.servers.ColorTransforms.ColorTransform
import qupath.opencv.ops.ImageOp
import qupath.opencv.ops.ImageOps
import qupath.lib.images.servers.PixelCalibration
import java.io.File
import static qupath.lib.gui.scripting.QPEx.*
import qupath.lib.gui.scripting.QPEx
import java.awt.image.BufferedImage;
import qupath.lib.common.ColorTools;
import qupath.lib.analysis.algorithms.EstimateStainVectors
import qupath.lib.color.ColorDeconvolutionHelper;
import qupath.lib.color.ColorDeconvolutionStains;
import qupath.lib.color.StainVector;
import qupath.lib.common.GeneralTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.plugins.parameters.ParameterList;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.RectangleROI;
import qupath.lib.gui.scripting.QPEx
import qupath.lib.roi.interfaces.ROI;
import qupath.ext.stardist.StarDist2D;
import qupath.lib.scripting.QP;
import qupath.lib.analysis.heatmaps.ColorModels
import qupath.lib.color.ColorMaps
import qupath.lib.gui.measure.ObservableMeasurementTableData

// SET OUTPUT
// def out_dir = buildFilePath('/local/data1/chrsp39/QuPath_Portable', 'results')
// mkdirs(out_dir)
def scriptDir = new File(getQuPath().getScriptPath()).getParent() // Get the script's directory
def out_dir = buildFilePath(scriptDir, "results") // Create a "results" folder in the script's directory
mkdirs(out_dir) // Create the directory if it doesn't exist

println "Output directory: ${out_dir}"

// Set up image:
setImageType('BRIGHTFIELD_H_DAB');
setColorDeconvolutionStains('{"Name" : "H-DAB default", "Stain 1" : "Hematoxylin", "Values 1" : "0.65111 0.70119 0.29049", "Stain 2" : "DAB", "Values 2" : "0.26917 0.56824 0.77759", "Background" : " 255 255 255"}');

// Create and select annotation excluding pen marks/dark areas with the predefined pixel classifier (Base_classifier):
resetSelection();
createAnnotationsFromPixelClassifier("Base_classifier", 20000.0, 8000.0, "DELETE_EXISTING", "INCLUDE_IGNORED")
selectAnnotations();

selectObjectsByClassification("Region*");

/////////////////////////////////////////// Compute stain normalization vectors /////////////////////////////////////////////
// Based on discussion in https://forum.image.sc/t/qupath-scripting-the-auto-stain-vector-estimation/40707/16.
ImageData<BufferedImage> imageData = QPEx.getCurrentImageData();
def imageName =  imageData.getServer().getMetadata().getName()
def imageNamePrefix = imageName.split("\\.")[0] // Save name to create the output file later 
print(String.format("Processing image: %s", imageNamePrefix))
println('Adjusting stains...')
if (imageData == null || !imageData.isBrightfield() || imageData.getServer() == null || !imageData.getServer().isRGB()) {
    DisplayHelpers.showErrorMessage("Estimate stain vectors", "No brightfield, RGB image selected!");
    return;
}
ColorDeconvolutionStains stains = imageData.getColorDeconvolutionStains();
if (stains == null || !stains.getStain(3).isResidual(

)) {
    DisplayHelpers.showErrorMessage("Estimate stain vectors", "Sorry, stain editing is only possible for brightfield, RGB images with 2 stains");
    return;
}
    
///////Select objects as a list///////   
cores_name = ['PathAnnotationObject']; 

//selectObjects = selectAnnotations();
cores_list = getAnnotationObjects();
//print(String.format("Selected objects: %s", cores_list.asList()))

//////////////////////////////////////////////////////////////////////////////////////////////
for (int i = 0; i < cores_list.size(); i++){ // There should only be one annotation, but left the loop just in case.
	
    PathObject pathObject = cores_list[i];
    ROI roi = pathObject == null ? null : pathObject.getROI();
    if (roi == null)
        roi = new RectangleROI(0, 0, imageData.getServer().getWidth(), imageData.getServer().getHeight());

    int MAX_PIXELS = 4000*4000;		
    double downsample = Math.max(1, Math.sqrt((roi.getBoundsWidth() * roi.getBoundsHeight())/ MAX_PIXELS));
    RegionRequest request = RegionRequest.createInstance(imageData.getServerPath(), downsample, roi);
    BufferedImage img = imageData.getServer().readBufferedImage(request);
    		
    // Apply small amount of smoothing to reduce compression artefacts
    img = EstimateStainVectors.smoothImage(img);
    // Check modes for background
    int[] rgb = img.getRGB(0, 0, img.getWidth(), img.getHeight(), null, 0, img.getWidth());
    int[] rgbMode = EstimateStainVectors.getModeRGB(rgb);
    int rMax = rgbMode[0];
    int gMax = rgbMode[1];
    int bMax = rgbMode[2];
    double minStain = 0.05;
    double maxStain = 1.0;
    double ignorePercentage = 1.0;

    ColorDeconvolutionStains stain_vec = EstimateStainVectors.estimateStains(img, stains, false)
        
    def hema_vec = stain_vec.getStain(1).toString()[1..-1].replace('ematoxylin: ','');
    def DAB_vec = stain_vec.getStain(2).toString()[1..-1].replace('AB: ','');
    def resi_vec = stain_vec.getStain(3).toString()[1..-1].replace('esidual: ','');
    def background_rgb = rgbMode.toString().replace(',','').replace('[','').replace(']','');   

// Save the first hema vector value to use later:
hema_vec_str = hema_vec[0..3]
hema_vec_num = hema_vec_str.toDouble() // Convert string to number

// Update the stain vectors:
setColorDeconvolutionStains('{"Name" : "H-DAB modified by script", "Stain 1" : "Hematoxylin", "Values 1" : "'+hema_vec+'", "Stain 2" : "DAB", "Values 2" : "'+DAB_vec+'", "Background" : " '+background_rgb+' "}');
}
print('Creating main annotation...')
def maxRed = imageData.getColorDeconvolutionStains().getMaxRed() // Get the background value

// Create a custom threshold to generate the final annotation, based on background value and hema vector (if necessary):
if (hema_vec_num >= 0.7) {
    new_Red = (maxRed - 1)
    general_norm = true
} else {
    new_Red = (maxRed - 7)
    general_norm = false
}

String th = new_Red.toString()

// Define the custom pixel classifier with the obtained threshold:
def tissue_json = """
{
  "pixel_classifier_type": "OpenCVPixelClassifier",
  "metadata": {
    "inputPadding": 0,
    "inputResolution": {
      "pixelWidth": {
        "value": 4.036,
        "unit": "µm"
      },
      "pixelHeight": {
        "value": 4.036,
        "unit": "µm"
      },
      "zSpacing": {
        "value": 1.0,
        "unit": "z-slice"
      },
      "timeUnit": "SECONDS",
      "timepoints": []
    },
    "inputWidth": 512,
    "inputHeight": 512,
    "inputNumChannels": 3,
    "outputType": "CLASSIFICATION",
    "outputChannels": [],
    "classificationLabels": {
      "0": {
        "name": "Tumor",
        "color": [
          200,
          0,
          0
        ]
      }
    }
  },
  "op": {
    "type": "data.op.channels",
    "colorTransforms": [
      {
        "channelName": "Red"
      }
    ],
    "op": {
      "type": "op.core.sequential",
      "ops": [
        {
          "type": "op.filters.gaussian",
          "sigmaX": 8.0,
          "sigmaY": 8.0
        },
        {
          "type": "op.threshold.constant",
          "thresholds": [
            """+th+"""
          ]
        }
      ]
    }
  }
}
"""

double minArea = 10000
double minHoleArea = 8000

def thresholder = GsonTools.getInstance().fromJson(tissue_json, qupath.lib.classifiers.pixel.PixelClassifier.class)

// Run the custom classifier to obtain and select the main annotation for the analysis:
createAnnotationsFromPixelClassifier(thresholder, minArea, minHoleArea, "DELETE_EXISTING")

selectObjectsByClassification("Region*");
clearSelectedObjects(true);
clearSelectedObjects();

selectAnnotations();

println "Detecting cells..."

/////////////////////////////////////////////// Option 1: DL with StarDist + thresholding //////////////////////////////////////////////////////////////////
// IMPORTANT! Replace this with the path to your StarDist model
// that takes 3 channel RGB as input (e.g. he_heavy_augment.pb)
// You can find some at https://github.com/qupath/models
// (Check credit & reuse info before downloading)
// def modelPath = "/local/data1/chrsp39/QuPath_Portable/models/he_heavy_augment.pb" //he_heavy_augment.pb" dsb2018

// get the model path
def scriptDir = new File(getQuPath().getScriptPath()).getParent() // Get the script's directory
def modelPath = buildFilePath(scriptDir, "models", "he_heavy_augment.pb") 

println "Using model path: ${modelPath}"

if(general_norm){ // Normalization over the downsampled full image.
  stardist = StarDist2D
    .builder(modelPath)
    .preprocess(
        ImageOps.Core.subtract(70),
        ImageOps.Core.divide(134)
    )
    //.normalizePercentiles(1, 99) // Percentile normalization
    .threshold(0.25)              // Probability (detection) threshold
    //.channels('DAB')
    .pixelSize(0.5)              // Resolution for detection 
    .ignoreCellOverlaps(true)
    .measureShape()              // Add shape measurements
    .measureIntensity()          // Add nucleus measurements
    .includeProbability(true)
    .doLog()
    .build()
}else{ // Relative normalization per tile
  stardist = StarDist2D
    .builder(modelPath)
    .normalizePercentiles(1, 99) // Percentile normalization
    .threshold(0.25)              // Probability (detection) threshold, 0.25
    //.channels('DAB') //////////// CHECK WHICH CHANNELS ARE AVAILABLE, DAB does not work
    .pixelSize(0.5)              // Resolution for detection
    .ignoreCellOverlaps(true)
    .measureShape()              // Add shape measurements
    .measureIntensity()          // Add nucleus measurements
    .includeProbability(true)
    .doLog()
    .build()
}

// Define which objects will be used as the 'parents' for detection
// Use QP.getAnnotationObjects() if you want to use all annotations, rather than selected objects
def pathObjects = QP.getSelectedObjects()
 
// Run detection for the selected objects
//def imageData = QP.getCurrentImageData()
if (pathObjects.isEmpty()) {
    QP.getLogger().error("No parent objects are selected!")
    return
}
stardist.detectObjects(imageData, pathObjects)
stardist.close() // This can help clean up & regain memory

///////////////////////////////// Object classification //////////////////////////////////////////////
// Script adapted from https://github.com/yau-lim/QuPath-Auto-Threshold/blob/main/autoThreshold.groovy
// @author Yau Mun Lim @yau-lim (2024)

println('Computing DAB threshold...')
/* PARAMETERS */
String channel = "DAB" // "HTX", "DAB", "Residual" for BF ; use channel name for FL ; "Average":Mean of all channels for BF/FL
double thresholdDownsample = 4 // 1:Full, 2:Very high, 4:High, 8:Moderate, 16:Low, 32:Very low, 64:Extremely low
def threshold = "Triangle" // Input threshold value for fixed threshold. Use the following for auto threshold: "Default", "Huang", "Intermodes", "IsoData", "IJ_IsoData", "Li", "MaxEntropy", "Mean", "MinError", "Minimum", "Moments", "Otsu", "Percentile", "RenyiEntropy", "Shanbhag", "Triangle", "Yen"
def thresholdFloor = null // Set a threshold floor value in case auto threshold is too low. Set null to disable
String output = "threshold value" // "annotation", "detection", "measurement", "preview", "threshold value"
// Reset preview overlay with "getQuPath().getViewer().resetCustomPixelLayerOverlay()"

double classifierDownsample = 4 // 1:Full, 2:Very high, 4:High, 8:Moderate, 16:Low, 32:Very low, 64:Extremely low
String classBelow = null // null or "Class Name"; use this for positive "Average" channel on brightfield
String classAbove = "Positive" // null or "Class Name"; use this for positive deconvoluted or fluorescence channels

/* Create object parameters */
//double minArea = 20000 // Already defined
//double minHoleArea = 5000 // Already defined

/* FUNCTIONS */
def autoThreshold(annotation, channel, thresholdDownsample, threshold, thresholdFloor, output, classifierDownsample, classBelow, classAbove, minArea, minHoleArea) {
    def qupath = getQuPath()
    def imageData = getCurrentImageData()
    def imageType = imageData.getImageType()
    def server = imageData.getServer()
    def cal = server.getPixelCalibration()
    def resolution = cal.createScaledInstance(classifierDownsample, classifierDownsample)
    def classifierChannel

    if (imageType.toString().contains("Brightfield")) {
        def stains = imageData.getColorDeconvolutionStains()

        if (channel == "HTX") {
            server = new TransformedServerBuilder(server).deconvolveStains(stains, 1).build()
            classifierChannel = ColorTransforms.createColorDeconvolvedChannel(stains, 1)
        } else if (channel == "DAB") {
            server = new TransformedServerBuilder(server).deconvolveStains(stains, 2).build()
            classifierChannel = ColorTransforms.createColorDeconvolvedChannel(stains, 2)
        } else if (channel == "Residual") {
            server = new TransformedServerBuilder(server).deconvolveStains(stains, 3).build()
            classifierChannel = ColorTransforms.createColorDeconvolvedChannel(stains, 3)
        } else if (channel == "Average") {
            server = new TransformedServerBuilder(server).averageChannelProject().build()
            classifierChannel = ColorTransforms.createMeanChannelTransform()
        }
    } else if (imageType.toString() == "Fluorescence") {
        if (channel == "Average") {
            server = new TransformedServerBuilder(server).averageChannelProject().build()
            classifierChannel = ColorTransforms.createMeanChannelTransform()
        } else {
            server = new TransformedServerBuilder(server).extractChannels(channel).build()
            classifierChannel = ColorTransforms.createChannelExtractor(channel)
        }
    } else {
        logger.error("Current image type not compatible with auto threshold.")
        return
    }

    // Check if threshold is Double (for fixed threshold) or String (for auto threshold)
    String thresholdMethod
    if (threshold instanceof String) {
        thresholdMethod = threshold
    } else {
        thresholdMethod = "Fixed"
    }

    // Apply the selected algorithm
    def validThresholds = ["Default", "Huang", "Intermodes", "IsoData", "IJ_IsoData", "Li", "MaxEntropy", "Mean", "MinError", "Minimum", "Moments", "Otsu", "Percentile", "RenyiEntropy", "Shanbhag", "Triangle", "Yen"]

    double thresholdValue
    if (thresholdMethod in validThresholds){
        // Determine threshold value by auto threshold method
        ROI pathROI = annotation.getROI() // Get QuPath ROI
        PathImage pathImage = IJTools.convertToImagePlus(server, RegionRequest.createInstance(server.getPath(), thresholdDownsample, pathROI)) // Get PathImage within bounding box of annotation
        def ijRoi = IJTools.convertToIJRoi(pathROI, pathImage) // Convert QuPath ROI into ImageJ ROI
        ImagePlus imagePlus = pathImage.getImage() // Convert PathImage into ImagePlus
        ImageProcessor ip = imagePlus.getProcessor() // Get ImageProcessor from ImagePlus
        ip.setRoi(ijRoi) // Add ImageJ ROI to the ImageProcessor to limit the histogram to within the ROI only
        
        /*
        if (darkBackground) {
            ip.setAutoThreshold("${thresholdMethod} dark")
        } else {*/
        ip.setAutoThreshold("${thresholdMethod}")
        //}

        thresholdValue = ip.getMaxThreshold()
        if (thresholdValue != null && thresholdValue < thresholdFloor) {
            thresholdValue = thresholdFloor
        }
    } else {
        logger.error("Invalid auto-threshold method")
        return
    }

    // If specified output is "threshold value, return threshold value in annotation measurements
    if (output == "threshold value") {
        logger.info("${thresholdMethod} threshold value: ${thresholdValue}")
        annotation.measurements.put("${thresholdMethod} threshold value", thresholdValue)
        return thresholdValue
    }
}

def annotations = getSelectedObjects().findAll{it.getPathClass() != getPathClass("Ignore*")}

if (annotations) {
    annotations.forEach{ anno ->
        thresholdValue = autoThreshold(anno, channel, thresholdDownsample, threshold, thresholdFloor, output, classifierDownsample, classBelow, classAbove, minArea, minHoleArea)//, classifierObjectOptions)
    }
} else {
    logger.warn("No annotations selected.")
}

// Use the obtained threshold to define a custom object classifier:

def object_json = """
{
  "object_classifier_type": "SimpleClassifier",
  "function": {
    "classifier_fun": "ClassifyByMeasurementFunction",
    "measurement": "DAB: Mean",
    "pathClassBelow": {
      "name": "Negative",
      "color": [
        112,
        112,
        225
      ]
    },
    "pathClassEquals": {
      "name": "Positive",
      "color": [
        250,
        62,
        62
      ]
    },
    "pathClassAbove": {
      "name": "Positive",
      "color": [
        250,
        62,
        62
      ]
    },
    "threshold": """+thresholdValue+"""
  },
  "pathClasses": [
    {
      "name": "Negative",
      "color": [
        112,
        112,
        225
      ]
    },
    {
      "name": "Positive",
      "color": [
        250,
        62,
        62
      ]
    }
  ],
  "filter": "DETECTIONS_ALL",
  "timestamp": 1729238139667
}
"""
println('Classifying cells...')

// Load and run the object classifier:
def obj_classifier = GsonTools.getInstance().fromJson(object_json, qupath.lib.classifiers.object.ObjectClassifier.class)
obj_classifier.classifyObjects(imageData, true)
fireHierarchyUpdate()

/////////////////////////////////////////////// Option 2: original QuPath positive detection //////////////////////////////////////////////////////////////////
// Uncomment the next lines and comment the lines in Option 1 to use the original workflow.
// // Run cell detection on WSI:
// // Relevant parameters:
// // - requestedPixelSizeMicrons: higher value reduces precision and running time.
// // - threshold: higher values demand a darker cell nuclei to be accepted (relevant if the tissue is darker in general)
// def ps = 0.5 // Pixel size variable. The lower it is the higher the computational demand
// runPlugin('qupath.imagej.detect.cells.PositiveCellDetection', '{"detectionImageBrightfield":"Hematoxylin OD","requestedPixelSizeMicrons":' + ps + ',"backgroundRadiusMicrons":8.0,"backgroundByReconstruction":true,"medianRadiusMicrons":0.0,"sigmaMicrons":1.5,"minAreaMicrons":10.0,"maxAreaMicrons":400.0,"threshold":0.05,"maxBackground":2.0,"watershedPostProcess":true,"cellExpansionMicrons":1.0,"includeNuclei":true,"smoothBoundaries":true,"makeMeasurements":true,"thresholdCompartment":"Nucleus: DAB OD mean","thresholdPositive1":0.2,"thresholdPositive2":0.4,"thresholdPositive3":0.6000000000000001,"singleThreshold":true}');


/////////////////////////////////////////////// Cell density map generation ///////////////////////////////////////////////////77
def radius = 100 as double // Search radius to create the density maps (higher radius = coarser heat map)

println "Preparing cell density maps..."
// DENSITY MAPPING
// use a predicate to determine which objects are included in the density mapping
def predicate = PathObjectPredicates.exactClassification(getPathClass('Negative'))
def builder = DensityMaps.builder(predicate) // Returns a DensityMapBuilder object

builder.radius(radius) // Set cell density radius

builder.buildClassifier(imageData) // to allow pixel size to be set according to input data
builder.type(DensityMaps.DensityMapType.SUM) // this is the default, setting anyway

fileName = buildFilePath(out_dir, imageNamePrefix + '_NegDMap' + '.tif')

println "saving negative cell density map..."

writeDensityMapImage(imageData, builder, fileName)

def predicate2 = PathObjectPredicates.exactClassification(getPathClass('Positive'))
def builder2 = DensityMaps.builder(predicate2) // Returns a DensityMapBuilder object

builder2.radius(radius) // Set cell density radius
 
builder2.buildClassifier(imageData) // to allow pixel size to be set according to input data
builder2.type(DensityMaps.DensityMapType.SUM) // this is the default, setting anyway

fileName2 = buildFilePath(out_dir, imageNamePrefix + '_PosDMap' + '.tif')

println "saving positive cell density map..."

writeDensityMapImage(imageData, builder2, fileName2)

///////////////////////////////////////////////// Read annotation measurements /////////////////////////////////////////////////////
println("Saving annotation data...")
def tissues = getAnnotationObjects()

def ob = new ObservableMeasurementTableData();
ob.setImageData(imageData, tissues);

// Define the table row names to access the data:
def imName = "Image"
def numDet = "Num Detections"
def area = "Area " + '\u00B5' + "m^2"
// Access the data:
tissues.each{tissue ->
    annotationImage = ob.getStringValue(tissue, imName)
    annotationDet = ob.getStringValue(tissue, numDet)
    annotationArea = ob.getStringValue(tissue, area)
}

// Read text file and load/update the annotation data:
def f = new File('Area_Det.txt')
contains = f.text.contains(annotationImage)
if (contains) {
    List data = f.readLines()
    f.text = ''
    data.each { line ->
        if (!line.contains(annotationImage)) {
            //log.info "${line}"
            f.text += line + "\n"
        }
    }
}
f.text += annotationImage + ';' + annotationDet + ';' + annotationArea + '\n'
println "Done!"

// The density maps are not normalized and can't be visualized directly,
// run the script summary_ratios.py to normalize them and gather statistics.