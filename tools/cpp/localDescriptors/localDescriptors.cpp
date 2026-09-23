#include <iostream>

#include <iomanip>
#include <limits>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/ply_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/concatenate.h>
#include <pcl/features/pfh.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/io.h>
#include <fstream>
#include <cmath>
#include <pcl/features/rops_estimation.h>
#include <pcl/features/usc.h>
#include <pcl/features/shot.h>
#include <pcl/features/fpfh.h>
#include <chrono>

//test edit
// now set up to take command line arguements


int main(int argc, char** argv)
{
	
	//timing
	std::chrono::steady_clock::time_point beginAll = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point beginSetup = std::chrono::steady_clock::now();
	
	//require the command line input
	if (argc != 7)
	{
		std::cerr << "Usage: localDescriptors <input.ply> <input.surfAreaFile> <output.outputCsv> <output.timeCsv> <input.alpha> <input.radiusRatio>" << std::endl;
		return 1;
	}
	
	//get arguements from command line
	//arguement 1 is input ply file
	//arguement 2 is a txt file with a single value that is the surface area of the ply
	//arguement 3 is output csv file for descriptors
	//arguement 4 is output csv file for times
	//arguement 5 is the alpha number for calculating the support radius as described by Zaharescu et al. (2012)
	//arguement 6 is a number describing the ratio of descriptor search radius to normal radius
	std::string inputFile = argv[1];
	std::string surfAreaFile = argv[2];
	std::string outputCsv = argv[3];
	std::string timeCsv = argv[4];
	double alpha = std::stod(argv[5]);
	double radiusRatio = std::stod(argv[6]);
	
	//check arguemnets
	if (radiusRatio <= 1) {
		std::cerr << "Error: radius ratio must be greater than 1 (descriptor radius must be greater than normal radius)." << std::endl;
		return 1;
	}
	
	
	//----------------------------------------------------------------------------
	//set descriptor and normal search radius
	//----------------------------------------------------------------------------
	
	// Read surface area from text file
	std::ifstream file(surfAreaFile);
	//check if surface area file can be opened
	if (!file) {
		std::cerr << "Error: could not open surface area file: " << surfAreaFile << std::endl;
		return 1;
	}
	//create variable for surface area
	double surfArea;
	//make sure the surface are can be read in
	//this if step reads in the file (file >> surfArea) but also errors out if it cant
	if(!(file >> surfArea)) {
		std::cerr << "Error: could not read surface area file: " << surfAreaFile << std::endl;
		return 1;
	}
	file.close();
	
	//calculate normal radius
	//get value of pi
	//following method from Zaharescu et al 2012 and used in Guo et al 2016
	//there are other methods for determining this region that use a point density informed distance, see Li et al 2016
	double pi = M_PI; //if this doesnt work there are other options for getting pi
	double areaFrac = (alpha * surfArea)/pi;
	double normRad = std::sqrt(areaFrac);
	
	//calculate descriptor radius
	//this must be larger than the normal radius
	double descrRad = normRad * radiusRatio;
	
	//timing
	std::chrono::steady_clock::time_point endSetup = std::chrono::steady_clock::now();
	
	//print surface area value
	std::cout << "Surface area: " << surfArea << std::endl;
	//print normal radius
	std::cout << "Normal search radius: " << normRad << std::endl;
	//print descriptor search radius
	std::cout << "Descriptor to normal search radius ratio: " << radiusRatio << std::endl;
	std::cout << "Descriptor search radius: " << descrRad << std::endl;
	
	
	
	
	//----------------------------------------------------------------------------
	//read in ply file
	//----------------------------------------------------------------------------
	
	//timing
	std::chrono::steady_clock::time_point beginRead = std::chrono::steady_clock::now();
	
	//new
	//create a polygonMesh object named mesh
	pcl::PolygonMesh mesh;
	
	//load ply into mesh and create error if cannot
	if (pcl::io::loadPLYFile(inputFile, mesh) == -1)
	{
		PCL_ERROR("Could not read input.ply\n");
		return -1;
	}
	
	//create empty point cloud object to store just point data from mesh
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	//extract point data from mesh
	pcl::fromPCLPointCloud2(mesh.cloud, *cloud);
	
	//timing
	std::chrono::steady_clock::time_point endRead = std::chrono::steady_clock::now();
	
	//output information
	std::cout << "Loaded " << cloud->size() << " points." << std::endl;
	std::cout << "Loaded " << mesh.polygons.size() << " polygons." << std::endl;
	
	//----------------------------------------------------------------------------
	//estimate normals
	//----------------------------------------------------------------------------
	
	//timing
	std::chrono::steady_clock::time_point beginNormals = std::chrono::steady_clock::now();
	
	// Create the normal estimation class, and pass the input dataset to it
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud (cloud);
	
	// Create an empty kdtree representation, and pass it to the normal estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
	ne.setSearchMethod (tree);
	
	// Output datasets
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
	
	// Use all neighbors in radius
	ne.setRadiusSearch (normRad);
	
	// Compute the features
	ne.compute (*cloud_normals);
	
	//timing
	std::chrono::steady_clock::time_point endNormals = std::chrono::steady_clock::now();
	
	// cloud_normals->size () should have the same size as the input cloud->size ()
	std::cout << "Computed " << cloud_normals->size() << " surface normals." << std::endl;
	
	//----------------------------------------------------------------------------
	//pfh features
	//----------------------------------------------------------------------------
	
	
	//timing
	std::chrono::steady_clock::time_point beginPfh = std::chrono::steady_clock::now();
	
	//following this tutorial, with augmentations to fit with what we have made previous
	//https://pointclouds.org/documentation/tutorials/pfh_estimation.html#pfh-estimation
	//there is a way to do the pfh features using a combined point and normal cloud but for now
	//i am doing it with the point and normal objects separate
	// Create the PFH estimation class, and pass the input dataset+normals to it
	pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
	pfh.setInputCloud (cloud);
	pfh.setInputNormals (cloud_normals);
	
	//use previously defined kd tree
	pfh.setSearchMethod (tree);
	
	// Output datasets
	pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs (new pcl::PointCloud<pcl::PFHSignature125> ());
	
	// Use all neighbors in radius
	// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
	pfh.setRadiusSearch (descrRad);
	
	// Compute the PFH features
	pfh.compute (*pfhs);
	
	//timing
	std::chrono::steady_clock::time_point endPfh = std::chrono::steady_clock::now();
	
	// output message about pfh features
	std::cout << "Computed " << pfhs->size() << " PFH features." << std::endl;
	
	//----------------------------------------------------------------------------
	//fpfh features
	//----------------------------------------------------------------------------
	
	//timing
	std::chrono::steady_clock::time_point beginFpfh = std::chrono::steady_clock::now();
	
	//follow these tutorials
	//https://pcl.readthedocs.io/projects/tutorials/en/master/fpfh_estimation.html#fpfh-estimation
	//https://robotica.unileon.es/index.php?title=PCL/OpenNI_tutorial_4:_3D_object_recognition_(descriptors)#FPFH
	
	//create estimation object 
	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
	fpfh.setInputCloud (cloud);
	fpfh.setInputNormals (cloud_normals);
	fpfh.setSearchMethod (tree);
	fpfh.setRadiusSearch (descrRad);
	
	//compute descriptors
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhFeats (new pcl::PointCloud<pcl::FPFHSignature33>());
	fpfh.compute (*fpfhFeats);
	
	//output is just a histogram like fph
	//https://pointclouds.org/documentation/structpcl_1_1_f_p_f_h_signature33.html
	
	//timing
	std::chrono::steady_clock::time_point endFpfh = std::chrono::steady_clock::now();
	
	//output message 
	std::cout << "Computed " << fpfhFeats->size() << " FPFH features." << std::endl;
	
	//----------------------------------------------------------------------------
	//RoPs features
	//----------------------------------------------------------------------------
	
	//timing
	std::chrono::steady_clock::time_point beginRops = std::chrono::steady_clock::now();
	
	//setting some hyperparams, using values following tutorial on pcl
	//https://pcl.readthedocs.io/projects/tutorials/en/master/rops_feature.html#rops-feature
	//these hyperparams are related to the number of features created (specified in estimation class)
	//though i am not 100% of the exact relationship
	unsigned int number_of_partition_bins = 5;
	unsigned int number_of_rotations = 3;
	
	// create RoPs estimateion class
	pcl::ROPSEstimation <pcl::PointXYZ, pcl::Histogram <135> > rops;
	//pass it argumenets
	//features calculated at every point so no need to supply setIndices
	rops.setSearchMethod (tree);
	rops.setRadiusSearch (descrRad);
	rops.setSupportRadius (descrRad);
	rops.setSearchSurface (cloud);
	rops.setInputCloud (cloud);
	rops.setTriangles (mesh.polygons);
	rops.setNumberOfPartitionBins (number_of_partition_bins);
	rops.setNumberOfRotations (number_of_rotations);
	
	//compute features
	pcl::PointCloud<pcl::Histogram <135> >::Ptr ropsFeats (new pcl::PointCloud <pcl::Histogram <135> > ());
	rops.compute (*ropsFeats);
	
	//timing
	std::chrono::steady_clock::time_point endRops = std::chrono::steady_clock::now();
	
	// output message about RoPs features
	std::cout << "Computed " << ropsFeats->size() << " RoPs features." << std::endl;
	
	
	//----------------------------------------------------------------------------
	//usc features
	//----------------------------------------------------------------------------
	
	//timing
	std::chrono::steady_clock::time_point beginUsc = std::chrono::steady_clock::now();
	
	//following tutorial: https://robotica.unileon.es/index.php?title=PCL/OpenNI_tutorial_4:_3D_object_recognition_(descriptors)
	//setting some hpyerparams
	//it also seems the pcl has some defaults? see "protected attributes" section of the api, but may be too large
	unsigned int minRadScale = 10;
	unsigned int densRadScale = 5;
	
	//create usc estimateion object
	pcl::UniqueShapeContext<pcl::PointXYZ, pcl::UniqueShapeContext1960, pcl::ReferenceFrame> usc;
	usc.setSearchMethod (tree);
	usc.setInputCloud (cloud);
	// Search radius, to look for neighbors. It will also be the radius of the support sphere.
	usc.setRadiusSearch (descrRad);
	// Set the radius to compute the Local Reference Frame.
	usc.setLocalRadius(descrRad);
	// The minimal radius value for the search sphere, to avoid being too sensitive
	// in bins close to the center of the sphere.
	usc.setMinimalRadius(descrRad / minRadScale);
	// Radius used to compute the local point density for the neighbors
	// (the density is the number of points within that radius).
	usc.setPointDensityRadius(descrRad / densRadScale);
	
	//compute features
	pcl::PointCloud<pcl::UniqueShapeContext1960>::Ptr uscFeats (new pcl::PointCloud<pcl::UniqueShapeContext1960>());
	usc.compute(*uscFeats);
	
	//note that the usc features have a different format than the previous features
	//you can see that info here: https://pointclouds.org/documentation/structpcl_1_1_unique_shape_context1960.html
	//it has one attribute named discriptor with 1960 length "vector" and an attribute named rf with 9 length "vector"
	//not entirely sure if the rf (reference frame) is important but outputting anyway
	
	//timing
	std::chrono::steady_clock::time_point endUsc = std::chrono::steady_clock::now();
	
	//output message
	std::cout << "Computed " << uscFeats->size() << " USC features and local reference frame." << std::endl;
	
	//----------------------------------------------------------------------------
	//shot descriptor
	//----------------------------------------------------------------------------
	
	//timing
	std::chrono::steady_clock::time_point beginShot = std::chrono::steady_clock::now();
	
	//following tutorial: https://robotica.unileon.es/index.php?title=PCL/OpenNI_tutorial_4:_3D_object_recognition_(descriptors)
	//create estimation object
	pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
	shot.setInputCloud (cloud);
	shot.setInputNormals (cloud_normals);
	shot.setSearchMethod (tree);
	shot.setRadiusSearch (descrRad);
	
	//compute features
	pcl::PointCloud<pcl::SHOT352>::Ptr shotFeats (new pcl::PointCloud<pcl::SHOT352>());
	shot.compute(*shotFeats);
	
	//similar to usc, the output from shot has a descriptor array of length 352 and a reference frame array of 9
	//https://pointclouds.org/documentation/structpcl_1_1_s_h_o_t352.html
	
	
	//timing
	std::chrono::steady_clock::time_point endShot = std::chrono::steady_clock::now();
	//output message
	std::cout << "Computed " << shotFeats->size() << " SHOT features and local reference frame." << std::endl;
	
	//----------------------------------------------------------------------------
	//output
	//----------------------------------------------------------------------------
	
	//timing
	std::chrono::steady_clock::time_point beginOutput = std::chrono::steady_clock::now();
	
	//there is a way to convert these features to a more flexible class called PCLPointCloud2
	//and concatenate them together but not using that right now, see commit: d6b87cd
	
	//begin output to csv file
	//number of features for each discriptor, this divides it by number of bytes to get the actual size of the object, not the size of the storage
	int nPfhs = sizeof(pfhs->points[0].histogram)/sizeof(pfhs->points[0].histogram[0]);
	int nFpfh = sizeof(fpfhFeats->points[0].histogram)/sizeof(fpfhFeats->points[0].histogram[0]);
	int nRopsFeats = sizeof(ropsFeats->points[0].histogram)/sizeof(ropsFeats->points[0].histogram[0]);
	int nUscFeats_desc = uscFeats->points[0].descriptorSize();
	int nUscFeats_rf = sizeof(uscFeats->points[0].rf)/sizeof(uscFeats->points[0].rf[0]);
	int nShotFeats_desc = sizeof(shotFeats->points[0].descriptor)/sizeof(shotFeats->points[0].descriptor[0]);
	int nShotFeats_rf = sizeof(shotFeats->points[0].rf)/sizeof(shotFeats->points[0].rf[0]);
	//open csv
	std::ofstream csv(outputCsv);
	if (!csv.is_open())
	{
		std::cerr << "Could not open csv output file: " << outputCsv << std::endl;
		return 1;
	}
	//make header
	csv << "x,y,z,nx,ny,nz";
	for (int i = 0; i < nPfhs; ++i)
	{
		csv << ",pfh_" << i + 1;
	}
	for (int i = 0; i < nFpfh; ++i)
	{
		csv << ",fpfh_" << i + 1;
	}
	for (int i = 0; i < nRopsFeats; ++i)
	{
		csv << ",rops_" << i + 1;
	}
	for (int i = 0; i < nUscFeats_desc; ++i)
	{
		csv << ",usc_desc_" << i + 1;
	}
	for (int i = 0; i < nUscFeats_rf; ++i)
	{
		csv << ",usc_rf_" << i + 1;
	}
	for (int i = 0; i < nShotFeats_desc; ++i)
	{
		csv << ",shot_desc_" << i + 1;
	}
	for (int i = 0; i < nShotFeats_rf; ++i)
	{
		csv << ",shot_rf_" << i + 1;
	}
	csv << "\n";
	//data
	//setting percision for the csv
	csv << std::setprecision(std::numeric_limits<float>::max_digits10);
	for (std::size_t i = 0; i < cloud->size(); ++i)
	{
		csv << cloud->points[i].x << ","
			<< cloud->points[i].y << ","
			<< cloud->points[i].z << ","
			<< cloud_normals->points[i].normal_x << ","
			<< cloud_normals->points[i].normal_y << ","
			<< cloud_normals->points[i].normal_z;
		for (int j = 0; j < nPfhs; ++j)
		{
			csv << "," << pfhs->points[i].histogram[j];
		}
		for (int j = 0; j < nFpfh; ++j)
		{
			csv << "," << fpfhFeats->points[i].histogram[j];
		}
		for (int j = 0; j < nRopsFeats; ++j)
		{
			csv << "," << ropsFeats->points[i].histogram[j];
		}
		for (int j = 0; j < nUscFeats_desc; ++j)
		{
			csv << "," << uscFeats->points[i].descriptor[j];
		}
		for (int j = 0; j < nUscFeats_rf; ++j)
		{
			csv << "," << uscFeats->points[i].rf[j];
		}
		for (int j = 0; j < nShotFeats_desc; ++j)
		{
			csv << "," << shotFeats->points[i].descriptor[j];
		}
		for (int j = 0; j < nShotFeats_rf; ++j)
		{
			csv << "," << shotFeats->points[i].rf[j];
		}
		csv << "\n";
	}
	//close csv
	csv.close();
	
	//timing
	std::chrono::steady_clock::time_point endOutput = std::chrono::steady_clock::now();
	
	std::cout << "Output csv saved to " << outputCsv << std::endl;
	//end output to csv
	
	//timing
	std::chrono::steady_clock::time_point endAll = std::chrono::steady_clock::now();
	
	
	//----------------------------------------------------------------------------
	//time differences
	//----------------------------------------------------------------------------
	
	//calculate differences
	std::chrono::seconds timeAll = std::chrono::duration_cast<std::chrono::seconds>(endAll - beginAll);
	std::chrono::seconds timeSetup = std::chrono::duration_cast<std::chrono::seconds>(endSetup - beginSetup);
	std::chrono::seconds timeRead = std::chrono::duration_cast<std::chrono::seconds>(endRead - beginRead);
	std::chrono::seconds timeNormals = std::chrono::duration_cast<std::chrono::seconds>(endNormals - beginNormals);
	std::chrono::seconds timePfh = std::chrono::duration_cast<std::chrono::seconds>(endPfh - beginPfh);
	std::chrono::seconds timeFpfh = std::chrono::duration_cast<std::chrono::seconds>(endFpfh - beginFpfh);
	std::chrono::seconds timeRops = std::chrono::duration_cast<std::chrono::seconds>(endRops - beginRops);
	std::chrono::seconds timeUsc = std::chrono::duration_cast<std::chrono::seconds>(endUsc - beginUsc);
	std::chrono::seconds timeShot = std::chrono::duration_cast<std::chrono::seconds>(endShot - beginShot);
	std::chrono::seconds timeOutput = std::chrono::duration_cast<std::chrono::seconds>(endOutput - beginOutput);
	
	//print times to console
	std::cout << "Total time: " << timeAll.count() << " seconds" << std::endl;
	std::cout << "Setup time: " << timeSetup.count() << " seconds" << std::endl;
	std::cout << "Read time: " << timeRead.count() << " seconds" << std::endl;
	std::cout << "Calculate normals: " << timeNormals.count() << " seconds" << std::endl;
	std::cout << "Calculate PFH: " << timePfh.count() << " seconds" << std::endl;
	std::cout << "Calculate FPFH: " << timeFpfh.count() << " seconds" << std::endl;
	std::cout << "Calculate RoPs: " << timeRops.count() << " seconds" << std::endl;
	std::cout << "Calculate USC: " << timeUsc.count() << " seconds" << std::endl;
	std::cout << "Calculate SHOT: " << timeShot.count() << " seconds" << std::endl;
	std::cout << "Output time: " << timeOutput.count() << " seconds" << std::endl;
	
	//output times to csv
	//open csv
	std::ofstream csv2(timeCsv);
	if (!csv2.is_open())
	{
		std::cerr << "Could not open time csv file: " << timeCsv << std::endl;
		return 1;
	}
	//make header
	csv2 << "task,timeSeconds";
	csv2 << "\n";
	//populate row by row
	csv2 << "total," << timeAll.count();
	csv2 << "\n";
	csv2 << "setup," << timeSetup.count();
	csv2 << "\n";
	csv2 << "read," << timeRead.count();
	csv2 << "\n";
	csv2 << "normals," << timeNormals.count();
	csv2 << "\n";
	csv2 << "pfh," << timePfh.count();
	csv2 << "\n";
	csv2 << "fpfh," << timeFpfh.count();
	csv2 << "\n";
	csv2 << "rops," << timeRops.count();
	csv2 << "\n";
	csv2 << "usc," << timeUsc.count();
	csv2 << "\n";
	csv2 << "shot," << timeShot.count();
	csv2 << "\n";
	csv2 << "output," << timeOutput.count();
	csv2 << "\n";
	//close csv
	csv2.close();
	
	std::cout << "Time csv saved to " << timeCsv << std::endl;
	
	
	return 0;
}
