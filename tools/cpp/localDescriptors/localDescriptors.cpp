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

//test edit
// now set up to take command line arguements


int main(int argc, char** argv)
{
	//require the command line input
	if (argc != 4)
	{
		std::cerr << "Usage: localDescriptors <input.ply> <output.ply> <output.csv>" << std::endl;
		return 1;
	}
	
	//get arguements from command line
	std::string inputFile = argv[1];
	std::string outputFile = argv[2];
	std::string outputCsv = argv[3];

	//this creates an empty point cloud object to store our point cloud in
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	
	// Read the PLY file
	//this bit is from chatgpt
	//previous hardcoded version just used "../../data/fileName" instead of inputFile
	if (pcl::io::loadPLYFile<pcl::PointXYZ>(inputFile, *cloud) == -1)
	{
		PCL_ERROR("Could not read input.ply\n");
		return -1;
	}
	
	std::cout << "Loaded " << cloud->size() << " points." << std::endl;
	
	// Create the normal estimation class, and pass the input dataset to it
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud (cloud);
	
	// Create an empty kdtree representation, and pass it to the normal estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
	ne.setSearchMethod (tree);
	
	// Output datasets
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
	
	// Use all neighbors in a sphere of radius 3cm
	ne.setRadiusSearch (0.03);
	
	// Compute the features
	ne.compute (*cloud_normals);
	
	// cloud_normals->size () should have the same size as the input cloud->size ()
	std::cout << "Computed " << cloud_normals->size() << " surface normals." << std::endl;
	
	//removing as concat comes later with PCLPointCloud2
	// combining points and normals into one ply for output
	//first step is initializing and empty point cloud to store this new set it
	//pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	//now combine original set with normals
	//pcl::concatenateFields(*cloud,*cloud_normals,*cloud_with_normals);
	
	//begin pfh features
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
	
	// Use all neighbors in a sphere of radius 5cm
	// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
	pfh.setRadiusSearch (0.05);
	
	// Compute the PFH features
	pfh.compute (*pfhs);
	
	// output message about pfh features
	std::cout << "Computed " << pfhs->size() << " pfh features." << std::endl;
	//finish pfh features
	
	
	//begin output to ply file
	//convert point cloud and extracted features into more flexible PCLPointCloud2 objects
	//initialize objects
	pcl::PCLPointCloud2 cloud_pcl;
	pcl::PCLPointCloud2 normals_pcl;
	pcl::PCLPointCloud2 pfhs_pcl;
	
	pcl::toPCLPointCloud2(*cloud, cloud_pcl);
	pcl::toPCLPointCloud2(*cloud_normals, normals_pcl);
	pcl::toPCLPointCloud2(*pfhs, pfhs_pcl);
	
	//combine points and normals
	pcl::PCLPointCloud2 cloudWithNormals_pcl;
	pcl::concatenateFields(cloud_pcl, normals_pcl, cloudWithNormals_pcl);
	
	//add pfh features
	pcl::PCLPointCloud2 featureDat_pcl;
	pcl::concatenateFields(cloudWithNormals_pcl, pfhs_pcl, featureDat_pcl);
	
	// now output as ply
	if (pcl::io::savePLYFile(outputFile, featureDat_pcl) == -1)
	{
		PCL_ERROR("Could not save output ply file\n");
		return 1;
	}

	//previous hardcode output
	//pcl::io::savePLYFile("../../data/remeshNormals.ply",*cloud_with_normals);

	//output message
	std::cout << "Saved point cloud with features to " << outputFile << std::endl;
	//finish output to ply file
	
	
	
	//begin output to csv file
	//number of features in pfhs
	int nPfhs = sizeof(pfhs->points[0].histogram)/sizeof(pfhs->points[0].histogram[0]);
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
	csv << "\n";
	//data
	//it is interesting that here we do not use the PCLPointCloud2 objects
	//i wonder if there is a reason not to, for now i am stinking with this
	
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
		csv << "\n";
	}
	//close csv
	csv.close();
	std::cout << "saved csv to " << outputCsv << std::endl;
	//end output to csv
	
	
	
	return 0;
}
