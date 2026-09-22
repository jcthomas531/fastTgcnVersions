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

//test edit
// now set up to take command line arguements


int main(int argc, char** argv)
{
	//require the command line input
	if (argc != 6)
	{
		std::cerr << "Usage: localDescriptors <input.ply> <output.csv> <input.surfAreaFile> <input.alpha> <input.radiusRatio>" << std::endl;
		return 1;
	}
	
	//get arguements from command line
	//arguement 1 is input ply file
	//arguement 2 is output csvFile
	//arguement 3 is a txt file with a single value that is the surface area of the ply
	//this could probably be improved
	//arguement 4 is the alpha number for calculating the support radius as described by Zaharescu et al. (2012)
	//arguement 5 is a number describing the ratio of descriptor search radius to normal radius
	std::string inputFile = argv[1];
	std::string outputCsv = argv[2];
	std::string surfAreaFile = argv[3];
	double alpha = std::stod(argv[4]);
	double radiusRatio = std::stod(argv[5]);
	
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
	//print surface area value
	std::cout << "Surface area: " << surfArea << std::endl;
	
	//calculate normal radius
	//get value of pi
	//following method from Zaharescu et al 2012 and used in Guo et al 2016
	//there are other methods for determining this region that use a point density informed distance, see Li et al 2016
	double pi = M_PI; //if this doesnt work there are other options for getting pi
	double areaFrac = (alpha * surfArea)/pi;
	double normRad = std::sqrt(areaFrac);
	//print normal radius
	std::cout << "Normal search radius: " << normRad << std::endl;
	
	//calculate descriptor radius
	//this must be larger than the normal radius
	double descrRad = normRad * radiusRatio;
	//print descriptor search radius
	std::cout << "Descriptor to normal search radius ratio: " << radiusRatio << std::endl;
	std::cout << "Descriptor search radius: " << descrRad << std::endl;
	
	
	
	
	//----------------------------------------------------------------------------
	//read in ply file
	//----------------------------------------------------------------------------
	
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
	//output information
	std::cout << "Loaded " << cloud->size() << " points." << std::endl;
	std::cout << "Loaded " << mesh.polygons.size() << " polygons." << std::endl;
	//end new
	
	
	
	//----------------------------------------------------------------------------
	//estimate normals
	//----------------------------------------------------------------------------
	
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
	
	// cloud_normals->size () should have the same size as the input cloud->size ()
	std::cout << "Computed " << cloud_normals->size() << " surface normals." << std::endl;
	
	
	//----------------------------------------------------------------------------
	//pfh features
	//----------------------------------------------------------------------------
	
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
	
	// output message about pfh features
	std::cout << "Computed " << pfhs->size() << " pfh features." << std::endl;
	
	
	
	
	//----------------------------------------------------------------------------
	//RoPs features
	//----------------------------------------------------------------------------
	
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
	
	// output message about pfh features
	std::cout << "Computed " << ropsFeats->size() << " RoPs features." << std::endl;
	
	
	//----------------------------------------------------------------------------
	//usc features
	//----------------------------------------------------------------------------
	
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
	
	//----------------------------------------------------------------------------
	//output
	//----------------------------------------------------------------------------
	
	//there is a way to convert these features to a more flexible class called PCLPointCloud2
	//and concatenate them together but not using that right now, see commit: d6b87cd
	
	//begin output to csv file
	//number of features for each discriptor, this divides it by number of bytes to get the actual size of the object, not the size of the storage
	int nPfhs = sizeof(pfhs->points[0].histogram)/sizeof(pfhs->points[0].histogram[0]);
	int nRopsFeats = sizeof(ropsFeats->points[0].histogram)/sizeof(ropsFeats->points[0].histogram[0]);
	int nUscFeats_desc = uscFeats->points[0].descriptorSize();
	int nUscFeats_rf = sizeof(uscFeats->points[0].rf)/sizeof(uscFeats->points[0].rf[0]);
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
		csv << "\n";
	}
	//close csv
	csv.close();
	std::cout << "saved csv to " << outputCsv << std::endl;
	//end output to csv
	
	
	
	return 0;
}
