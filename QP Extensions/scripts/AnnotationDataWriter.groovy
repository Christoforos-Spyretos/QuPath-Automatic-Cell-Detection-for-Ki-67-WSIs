import qupath.lib.gui.measure.ObservableMeasurementTableData
import java.io.File

// Select the current image and annotation:
def imageData = getCurrentImageData()
def tissues = getAnnotationObjects()

// Get the annotation measurement table:
def ob = new ObservableMeasurementTableData();
ob.setImageData(imageData, tissues);

// Define the names of the row to access the data:
def imName = "Image"
def numDet = "Num Detections"
def area = "Area " + '\u00B5' + "m^2"

// Access the data with the row names:
tissues.each{tissue ->
    annotationImage = ob.getStringValue(tissue, imName)
    annotationDet = ob.getStringValue(tissue, numDet)
    annotationArea = ob.getStringValue(tissue, area)
}

// Read the text file with the saved annotation data:
textFile = 'Area_Det.txt'
println("Reading annotation from " + annotationImage + "...")

f = new File(textFile)

// Load information for the current slide if not saved already:
contains = f.text.contains(annotationImage)
if(!contains) {
    println("Annotation area: " + annotationArea + " \u00B5" + 'm^2')
    println("Number of cells detected: " + annotationDet)
    f.text += annotationImage + ';' + annotationDet + ';' + annotationArea + '\n'
    println('Annotation data saved!')
} else {
   println("Annotation data already in file, skipping...") 
}
