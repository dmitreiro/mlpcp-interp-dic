# -*- coding: mbcs -*-
# Do not delete the following import lines
#abaqus cae noGUI=Cruciform.py
#This is the 3D Cruciform NM100 Mesh
os.system('Abaqus job=Job-1.inp user=UMMDp_FLC.f interactive ask_delete=OFF')
from abaqus import *
from abaqusConstants import *
# from odbAccess import *
from pyD import new
from array import array

import __main__
import glob
import os
import csv
import time
import  sys
import numpy as np
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior

# Define path for data
current_dir = os.getcwd()
# MYCSVDIR = os.path.join(current_dir, 'data', 'regular_mesh')
DIC_DIR = os.path.join(current_dir, 'data', 'dic')
TRAIN_DIR = os.path.join(current_dir, 'data', 'dic', 'train_samples')
TEST_DIR = os.path.join(current_dir, 'data', 'dic', 'test_samples')

# normalize paths
# MYCSVDIR = os.path.normpath(MYCSVDIR)
DIC_DIR = os.path.normpath(DIC_DIR)
TRAIN_DIR = os.path.normpath(TRAIN_DIR)
TEST_DIR = os.path.normpath(TEST_DIR)

# create directories if not exist
# os.makedirs(MYCSVDIR, exist_ok=True)
os.makedirs(DIC_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

overwrite=True

#  ----------------------------------------------------------------------------------------------------------------  #
# -----------------------------------  Define the Extrusion Part and Partition  -----------------------------------  #
#  ----------------------------------------------------------------------------------------------------------------  #

s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.Line(point1=(0.0, 0.0), point2=(30.0, 0.0))
s.HorizontalConstraint(entity=g[2], addUndoState=False)
s.Line(point1=(30.0, 0.0), point2=(30.0, 15.0))
s.VerticalConstraint(entity=g[3], addUndoState=False)
s.PerpendicularConstraint(entity1=g[2], entity2=g[3], addUndoState=False)
s.Line(point1=(0.0, 0.0), point2=(0.0, 30.0))
s.VerticalConstraint(entity=g[4], addUndoState=False)
s.PerpendicularConstraint(entity1=g[2], entity2=g[4], addUndoState=False)
s.Line(point1=(0.0, 30.0), point2=(15.0, 30.0))
s.HorizontalConstraint(entity=g[5], addUndoState=False)
s.PerpendicularConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
s.FixedConstraint(entity=v[0])
s.EqualLengthConstraint(entity1=g[2], entity2=g[4])
s.EqualLengthConstraint(entity1=g[3], entity2=g[5])
s.CircleByCenterPerimeter(center=(15.0, 15.0), point1=(22.5, 23.75))
s.Line(point1=(15.0, 30.0), point2=(15.0, 26.5244305715896))
s.VerticalConstraint(entity=g[7], addUndoState=False)
s.PerpendicularConstraint(entity1=g[5], entity2=g[7], addUndoState=False)
s.CoincidentConstraint(entity1=v[7], entity2=g[6], addUndoState=False)
s.Line(point1=(30.0, 15.0), point2=(26.5244305715896, 15.0))
s.HorizontalConstraint(entity=g[8], addUndoState=False)
s.PerpendicularConstraint(entity1=g[3], entity2=g[8], addUndoState=False)
s.CoincidentConstraint(entity1=v[8], entity2=g[6], addUndoState=False)
s.autoTrimCurve(curve1=g[6], point1=(19.2347831726074, 25.6309909820557))
s.autoTrimCurve(curve1=g[9], point1=(24.0811347961426, 21.9118900299072))
s.FixedConstraint(entity=v[11])
s.RadialDimension(curve=g[10], textPoint=(3.96877670288086, 3.67629909515381), 
    radius=7.0)
s.FilletByRadius(radius=2.5, curve1=g[7], nearPoint1=(14.8730659484863, 
    23.1115970611572), curve2=g[10], nearPoint2=(13.4191551208496, 
    21.7919178009033))
s.FilletByRadius(radius=2.5, curve1=g[8], nearPoint1=(22.9907035827637, 
    14.9535703659058), curve2=g[10], nearPoint2=(22.1425971984863, 
    13.7538595199585))
s.ObliqueDimension(vertex1=v[3], vertex2=v[4], textPoint=(8.93628692626953, 
    38.5878562927246), value=15.0)
s.dragEntity(entity=v[14], points=((12.5, 24.1651513899117), (12.5, 23.75), (
    13.75, 26.25), (13.75, 27.5)))
s.dragEntity(entity=v[4], points=((15.0, 30.0), (15.0, 30.0), (16.25, 30.0), (
    13.75, 30.0), (11.25, 28.75)))
s.undo()
s.ObliqueDimension(vertex1=v[0], vertex2=v[1], textPoint=(14.1344108581543, 
    -7.11508655548096), value=30.0)
p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=THREE_D, 
    type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['Part-1']
p.BaseSolidExtrude(sketch=s, depth=0.5)
s.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['Model-1'].sketches['__profile__']
p = mdb.models['Model-1'].parts['Part-1']
f, e, d1 = p.faces, p.edges, p.datums
t = p.MakeSketchTransform(sketchPlane=f[9], sketchUpEdge=e[19], 
    sketchPlaneSide=SIDE1, origin=(12.169049, 12.169049, 0.5))
s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
    sheetSize=84.85, gridSpacing=2.12, transform=t)
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.setPrimaryObject(option=SUPERIMPOSE)
p = mdb.models['Model-1'].parts['Part-1']
p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
s1.rectangle(point1=(-12.169049, -12.169049), point2=(-2.11879614980537, 
    -2.11879678680635))
s1.CoincidentConstraint(entity1=v[13], entity2=g[5], addUndoState=False)
s1.EqualDistanceConstraint(entity1=v[3], entity2=v[5], midpoint=v[13], 
    addUndoState=False)
p = mdb.models['Model-1'].parts['Part-1']
f = p.faces
pickedFaces = f.getSequenceFromMask(mask=('[#200 ]', ), )
e1, d2 = p.edges, p.datums
p.PartitionFaceBySketch(sketchUpEdge=e1[19], faces=pickedFaces, sketch=s1)
s1.unsetPrimaryObject()
del mdb.models['Model-1'].sketches['__profile__']
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)

#  ----------------------------------------------------------------------------------------------------------------  #
# -------------------------------------------  Define the Material  -----------------------------------------------  #
#  ----------------------------------------------------------------------------------------------------------------  #

mdb.models['Model-1'].Material(name='Material-1')
mdb.models['Model-1'].materials['Material-1'].Depvar(n=7)
mdb.models['Model-1'].materials['Material-1'].UserOutputVariables(n=6)
mdb.models['Model-1'].materials['Material-1'].UserMaterial(
mechanicalConstants=())
mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', 
    material='Material-1', thickness=None)                                                                                                                                                                            #Section Homogeneous Solid
p = mdb.models['Model-1'].parts['Part-1']
c = p.cells
cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
region = p.Set(cells=cells, name='Set-1')
p = mdb.models['Model-1'].parts['Part-1']
p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', 
    thicknessAssignment=FROM_SECTION)                                                                                                                                                                                               #Section Assignment
p = mdb.models['Model-1'].parts['Part-1']
c = p.cells
cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
region = regionToolset.Region(cells=cells)
orientation=None
mdb.models['Model-1'].parts['Part-1'].MaterialOrientation(region=region, 
    orientationType=SYSTEM, axis=AXIS_3, localCsys=orientation, 
    fieldName='', additionalRotationType=ROTATION_ANGLE, 
    additionalRotationField='', angle=0.0, stackDirection=STACK_3)                                                                                                                         #Alfa Defining for Rotation of Field
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
a = mdb.models['Model-1'].rootAssembly
a.DatumCsysByDefault(CARTESIAN)
p = mdb.models['Model-1'].parts['Part-1']
a.Instance(name='Part-1-1', part=p, dependent=ON)                                                                                                                                                             #Assembly Instance
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON)
mdb.models['Model-1'].StaticStep(name='Step-1', previous='Initial', 
    maxNumInc=100000, initialInc=0.1, minInc=1e-20, nlgeom=ON)                                                                                                                               #Step 1 Define | nlgeo ON | increment
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
    'CDISP', 'CF', 'CSTRESS', 'LE', 'PE', 'PEEQ', 'PEMAG', 'RF', 'S', 'TF', 
    'U','SDV','UVARM', 'COORD'), numIntervals=20)                                                                                                                                                            #Field Output Defining SDV UVARM LE TF 20 FRAMES
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON, adaptiveMeshConstraints=OFF)
a = mdb.models['Model-1'].rootAssembly
f1 = a.instances['Part-1-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#100 ]', ), )

#  ----------------------------------------------------------------------------------------------------------------  #
# -----------------------------------------  Boundary Conditions Defining  ----------------------------------------  #
#  ----------------------------------------------------------------------------------------------------------------  #

region = a.Set(faces=faces1, name='Set-1')                                                                                                                                                                       #U1 Belongs on Set-2 Wich is x direction
mdb.models['Model-1'].DisplacementBC(name='BC-1', createStepName='Step-1', 
    region=region, u1=2.0, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0, 
    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
    localCsys=None)                                                                                                                                                                                                                                 #U1 = 2 mm
a = mdb.models['Model-1'].rootAssembly
f1 = a.instances['Part-1-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#4 ]', ), )
region = a.Set(faces=faces1, name='Set-2')                                                                                                                                                                       #U2 Belongs on Set-2 Wich is y direction
mdb.models['Model-1'].DisplacementBC(name='BC-2', createStepName='Step-1', 
    region=region, u1=0.0, u2=2.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0, 
    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
    localCsys=None)                                                                                                                                                                                                                                 #U2 = 2 mm
a = mdb.models['Model-1'].rootAssembly
f1 = a.instances['Part-1-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#400 ]', ), )
region = a.Set(faces=faces1, name='Set-3')                                                                                                                                                                       #Set-3 is the X Symetry BC (left)
mdb.models['Model-1'].XsymmBC(name='BC-3', createStepName='Step-1', 
    region=region, localCsys=None)
a = mdb.models['Model-1'].rootAssembly
f1 = a.instances['Part-1-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#200 ]', ), )
region = a.Set(faces=faces1, name='Set-4')                                                                                                                                                                       #Set-4 is the Y Symetry BC (down)
mdb.models['Model-1'].YsymmBC(name='BC-4', createStepName='Step-1', 
    region=region, localCsys=None)
a = mdb.models['Model-1'].rootAssembly
f1 = a.instances['Part-1-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#1000 ]', ), )
region = a.Set(faces=faces1, name='Set-5')                                                                                                                                                                       #Set-5 is the Z Symetry BC (behind)
mdb.models['Model-1'].ZsymmBC(name='BC-5', createStepName='Step-1', 
    region=region, localCsys=None)
    
#  ----------------------------------------------------------------------------------------------------------------  #
# -----------------------------------------------  Mesh Defining  -------------------------------------------------  #
#  ----------------------------------------------------------------------------------------------------------------  #

session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF, 
    bcs=OFF, predefinedFields=OFF, connectors=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=ON)
p = mdb.models['Model-1'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
    engineeringFeatures=OFF, mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
p = mdb.models['Model-1'].parts['Part-1']
p.seedPart(size=1.0, deviationFactor=0.1, minSizeFactor=0.1)
p = mdb.models['Model-1'].parts['Part-1']
c = p.cells
pickedRegions = c.getSequenceFromMask(mask=('[#1 ]', ), )
p.setMeshControls(regions=pickedRegions, algorithm=MEDIAL_AXIS)
elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD, 
    kinematicSplit=AVERAGE_STRAIN, secondOrderAccuracy=OFF, 
    hourglassControl=ENHANCED, distortionControl=DEFAULT)
elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)
p = mdb.models['Model-1'].parts['Part-1']
c = p.cells
cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
pickedRegions =(cells, )
p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
    elemType3))
p = mdb.models['Model-1'].parts['Part-1']
p.generateMesh()
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
a1 = mdb.models['Model-1'].rootAssembly
a1.regenerate()
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=OFF)

#  ----------------------------------------------------------------------------------------------------------------  #
# ------------------------------------------------  Job 1  ---------------------------------------------------------  #
#  ----------------------------------------------------------------------------------------------------------------  #

mdb.Job(name='Job-1', model='Model-1', description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
    numGPUs=0)
mdb.jobs['Job-1'].submit(consistencyChecking=OFF)
    
#  ----------------------------------------------------------------------------------------------------------------  #
# --------------------------------------  Define the Design of Experiments  ---------------------------------------  #
#  ----------------------------------------------------------------------------------------------------------------  #

os.system('Abaqus job=Job-1.inp user=UMMDp_FLC.f interactive ask_delete=OFF')
doe = new( [0.600, 0.600, 0.600, 280.00, 120.00, 0.100], [6.000, 6.000, 6.000, 700.00, 300.00, 0.300], 6, 6000)
#TODO Replace this by own loop for parameters

for index, valor in enumerate(doe):
    valor_r0=round(float(valor[0]),3)
    valor_r45=round(float(valor[1]),3)
    valor_r90=round(float(valor[2]),3)
    valor_k=round(float(valor[3]),2)
    valor_sigma0=round(float(valor[4]),2)
    valor_n=round(float(valor[5]),3)

    valor_F = round(float(valor_r0 /(valor_r90 * (valor_r0 + 1))),4)
    valor_G = round(float(1 /(valor_r0 + 1)),4) 
    valor_H = round(float(1-valor_G),4)
    valor_L = 1.5
    valor_M = 1.5
    valor_N = round(float(0.5 * (((valor_r0 + valor_r90) * (2 * valor_r45 + 1)) / (valor_r90 * (valor_r0 + 1)))),4)
    valor_e0= round(float((valor_sigma0/valor_k)**(1/valor_n)),4)
    valor_E=210000
    valor_v=0.3

    mdb.models['Model-1'].Material(name='Material-1')
    mdb.models['Model-1'].materials['Material-1'].Depvar(n=7)
    mdb.models['Model-1'].materials['Material-1'].UserOutputVariables(n=15)
    mdb.models['Model-1'].materials['Material-1'].UserMaterial(mechanicalConstants=(0, 0, valor_E, valor_v, 1, valor_F, valor_G, valor_H, valor_L, valor_M, valor_N, 2, valor_k, valor_e0, valor_n, 0))
    mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', 
    material='Material-1', thickness=None)
    
    #  ----------------------------------------------------------------------------------------------------------------  #
    # ------------------------------------------------  Job 2  ---------------------------------------------------------  #
    #  ----------------------------------------------------------------------------------------------------------------  #

    mdb.Job(name='Job-1', model='Model-1', description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
        numGPUs=0)
    mdb.jobs['Job-1'].submit(consistencyChecking=OFF)                                                                                                                                                             #Submit the Job
    mdb.jobs['Job-1'].waitForCompletion()
    
    os.system('Abaqus job=Job-1.inp user=UMMDp_FLC.f interactive ask_delete=OFF')
    
    odb_Path = 'Job-1.odb'
    cruci_ODB = session.openOdb(name=odb_Path)                                                                                                                                                                            #Open ODB File

    frames = len(cruci_ODB.steps['Step-1'].frames)                                                                                                                                                                #Number of Frames
    
    nodes = len(cruci_ODB.steps['Step-1'].frames[0].fieldOutputs['S'].values[0].instance.nodes)                                                                #Number of Nodes

    elements = len(cruci_ODB.steps['Step-1'].frames[0].fieldOutputs['S'].values[0].instance.elements)                                                  #Number of Elements
   
    no_of_field_output_ut_values = len(cruci_ODB.steps['Step-1'].frames[0].fieldOutputs['S'].values)                                                  #Field Output Values
    

    #  ----------------------------------------------------------------------------------------------------------------  #
    # ---------------------------------------  Write Values for Data Base  --------------------------------------------  #
    #  ----------------------------------------------------------------------------------------------------------------  #

    #TODO 

    #  ----------------------------------------------------------------------------------------------------------------  #
    # ---------------------------------------  Write Values for Data Base  --------------------------------------------  #
    #  ----------------------------------------------------------------------------------------------------------------  #

    # Define CSV file path
    csv_filename = r'{}\{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(MYCSVDIR, valor_F, valor_G, valor_H, valor_L, valor_M, valor_N, valor_sigma0, valor_k, valor_n)

    # Open the CSV file in binary mode ('wb')
    with open(csv_filename, 'wb') as file:
        writer = csv.writer(file)
    
        # Create a list to store all rows of data
        all_data = []
    
        # Loop for Frames
        for i in range(1, frames):
            data = []
            somax = 0  
            somay = 0
            nodes_x = [2, 5, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 635, 638, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685]  # Nodes in Set-1 X are n-1 inside nodes_x[} 
            nodes_y = [9, 11, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 642, 644, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767]  # Nodes in Set-2 Y are n-1 inside nodes_y[} 
     
           # Calculate somax and somay
            for nx in nodes_x:
                somax += abs(cruci_ODB.steps['Step-1'].frames[i].fieldOutputs['TF'].values[nx].data[0])
            
            for ny in nodes_y:
                somay += abs(cruci_ODB.steps['Step-1'].frames[i].fieldOutputs['TF'].values[ny].data[1])
            
            # Append somax and somay to data
            data.extend([somax, somay])
            
            # Calculate other values and append to data
            for ii in range(no_of_field_output_ut_values):
                node_displacement = cruci_ODB.steps['Step-1'].frames[i].fieldOutputs['S'].values[ii].data[0]
                strainx = cruci_ODB.steps['Step-1'].frames[i].fieldOutputs['LE'].values[ii].data[0]
                strainy = cruci_ODB.steps['Step-1'].frames[i].fieldOutputs['LE'].values[ii].data[1]
                strainxy = cruci_ODB.steps['Step-1'].frames[i].fieldOutputs['LE'].values[ii].data[3]
                
                data.extend([strainx, strainy, strainxy])
            
            # Append the row data to all_data
            all_data.append(data)
        
        # Use writerows() to write all_data to the CSV file in bulk
        writer.writerows(all_data)

    if index == 0:
        # Define CSV file path
        csv_nodes = r'{}\nodes.csv'.format(MYCSVDIR)

        # Open a .csv file to write the results
        with open(csv_nodes, 'wb') as f_nodes:
            nodes_writer = csv.writer(f_nodes)
            
            # Create a list to store all rows of data
            nodes_all_data = []
            
            # Loop for nodes and coordinates in all frames
            # # Loop for frames
            # for frame in range(frames):
            #     nodes_data = []
            #     # Loop for nodes
            #     for node in range(nodes):
            #         n_cords = cruci_ODB.steps['Step-1'].frames[frame].fieldOutputs['COORD'].values[node].data
            #         # Loop for coordinates
            #         for cord in n_cords:
            #             nodes_data.extend([cord])

            #     # Append the row data to all_data
            #     nodes_all_data.append(nodes_data)

            # Loop for nodes and coordinates only in frame 0
            nodes_data = []
            # Loop for nodes
            for node in range(nodes):
                n_cords = cruci_ODB.steps['Step-1'].frames[0].fieldOutputs['COORD'].values[node].data
                # Loop for coordinates
                for cord in n_cords:
                    nodes_data.extend([cord])

            # Append the row data to all_data
            nodes_all_data.append(nodes_data)

            # Use writerows() to write all_data to the CSV file in bulk
            nodes_writer.writerows(nodes_all_data)
        
        # Define CSV file path
        csv_elements = r'{}\elements.csv'.format(MYCSVDIR)

        with open(csv_elements, 'wb') as f_elements:
            el_writer = csv.writer(f_elements)

            # Create a list to store all rows of data
            el_all_data = []

            # Loop for elements
            for el in range(elements):
                el_data = []
                el_nodes = cruci_ODB.steps['Step-1'].frames[0].fieldOutputs['S'].values[0].instance.elements[el].connectivity
                for n in el_nodes:
                    el_data.extend([n])

                # Append the row data to all_data
                el_all_data.append(el_data)

            # Use writerows() to write all_data to the CSV file in bulk
            el_writer.writerows(el_all_data)

        inst_el = a.instances['Part-1-1'].elements
        csv_centroids = r'{}\centroids.csv'.format(MYCSVDIR)

        # Open a .csv file to write the results
        with open(csv_centroids, 'wb') as f_cent:
            cent_writer = csv.writer(f_cent)
            
            # Create a list to store all rows of data
            centroid_list = []
            
            # Loop for elements
            for i, el in enumerate(inst_el):
                region = regionToolset.Region(elements=inst_el[i:i+1])
                properties = a.getMassProperties(regions=region)

                # Append the row data
                centroid_list.append(list(properties['volumeCentroid']))

            # Use writerows() to write all_data to the CSV file in bulk
            cent_writer.writerows(centroid_list)

    # Close the ODB file
    cruci_ODB.close()