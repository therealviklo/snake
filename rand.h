#pragma once
#include <random>
#include <limits>

int randi();
unsigned randu();
size_t randz();
double rand01();
double rand11();
double randd(double min, double max);
double randnorm(double mean, double stddev);
double xavier(size_t nin);
double hzrs(size_t nin);