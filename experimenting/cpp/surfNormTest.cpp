#include <iostream>

#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/ply_io.h>
#include <pcl/search/kdtree.h>

int main()
{
	//this creates an empty point cloud object to store our point cloud in
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	
	// Read the PLY file
	//this bit is from chatgpt
	//must change input file before running
	if (pcl::io::loadPLYFile<pcl::PointXYZ>("input.ply", *cloud) == -1)
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
}