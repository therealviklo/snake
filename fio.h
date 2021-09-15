#pragma once
#include <fstream>

template <class T>
void writeToFile(const T& t, const char* fname)
{
	std::ofstream fs;
	fs.exceptions(std::ios::badbit | std::ios::failbit);
	fs.open(fname, std::ios::binary);
	fs.write((const char*)&t, sizeof(t));
}

template <class T>
void readFromFile(T& t, const char* fname)
{
	std::ifstream fs;
	fs.exceptions(std::ios::badbit | std::ios::failbit);
	fs.open(fname, std::ios::binary);
	fs.read((char*)&t, sizeof(t));
}