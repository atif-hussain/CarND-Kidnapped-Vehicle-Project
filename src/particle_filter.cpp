/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <set>
#include <tuple>

#include "particle_filter.h"

using namespace std;

// the random engine to use can be created only once
static default_random_engine gen;

int countUnique(vector<Particle>& particles) {
	std::set<tuple<double,double,double>> ids;
	for (const auto& p : particles)
			ids.insert(std::make_tuple(p.x,p.y,p.theta));
	return ids.size();
}

void logParticles(vector<Particle>& particles, bool top10=false) {
	if(top10) std::sort(particles.begin(), particles.end(), 
		[] (const Particle& lhs, const Particle& rhs) {    return lhs.weight > rhs.weight; });
	for (int i=0; i< (top10?10:particles.size()); i++) {
		auto&& p = particles[i];
		cout << "(" << p.id << "," << p.x << "," << p.y << "," << p.theta << "," << p.weight << "), \t";
	} cout << endl << endl;
}

void ParticleFilter::init(double gps_x, double gps_y, double compass_theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	//input: gps' x, gps' y, initial heading estimate theta, std = array of uncertainties for these measurements
	// sample from a Gaussian distribution, centred around these measurements to initialize all parameters
		//std::random_device rd;
		//std::mt19937 gen(rd());
	std::normal_distribution<double> xn(gps_x, std[0]);
	std::normal_distribution<double> yn(gps_y, std[1]);
	std::normal_distribution<double> zn(compass_theta, std[2]);

	int N = 50; 
	num_particles = N;
	particles.resize(N);
	cout << "initializing " << N << " particles, around (" <<gps_x<<","<<gps_y<<", @"<<compass_theta<<"angle)"  << endl;
	for (int i=0; i<N; i++) {
		auto&& p = particles[i];
		p.id = i;
		p.x = xn(gen);		//add random gaussian noise
		p.y = yn(gen); 	//add random gaussian noise 
		p.theta = zn(gen); //add random gaussian noise  rand()*2*std::PPI;
		
		// weights vector is not initialized
		// particles[].weight are kept non-normalized till the point they can be
		p.weight = 1.0;
	}
	logParticles(particles);
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
		//std::random_device rd;
		//std::mt19937 gen(rd());
	std::normal_distribution<> std_noise(0.0, 1.0);
	std_noise(gen); // give random gaussian noise

	if (fabs(yaw_rate) < 0.0001) yaw_rate = 0.0001;

	cout << "Predicted " << " particles... " ;
	for (Particle& p: particles) {
		auto new_theta = p.theta + yaw_rate * delta_t;
		p.x = p.x + velocity/yaw_rate * (std::sin(new_theta)-std::sin(p.theta) ) + std_noise(gen) * std_pos[0]; 
		p.y = p.y - velocity/yaw_rate * (std::cos(new_theta)-std::cos(p.theta) ) + std_noise(gen) * std_pos[1];
		p.theta = new_theta + std_noise(gen) * std_pos[2];
	}
//	logParticles(particles);
}

inline double sq(double x) { return x*x; }
static double by2PI = 1.0 / std::sqrt(2*M_PI);
double norm_prob(double z) {
	double prob = exp( - sq(z) / 2 ) * by2PI ;
	return prob;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
//	cout << "Data associations :" << endl; 
	for (LandmarkObs& obs: observations) {
		double minD2 = 1000000.0; LandmarkObs* closest = &predicted[0];
		for (LandmarkObs& pred: predicted) {
			double d = sq(pred.x-obs.x) + sq(pred.y-obs.y) ;
			if (d < minD2) { minD2 = d; closest = &pred; }
		}
		obs.id = closest->id;
//		cout << "obs=" << obs.x << "," << obs.y << " has nearest landmark " << closest->x << "," << closest->y << " at dist=" << dist(obs.x,obs.y,closest->x,closest->y) << endl;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
///	cout << "Updating Weights based on " << observations.size() << " observations..." << endl;
///	for(auto&& obs:observations)
///		cout << "\t (" << obs.id << "," << obs.x << "," << obs.y << "), ";
///	cout << endl;
	
    // Iterate over all particles
    for (auto&& p : particles) {
	
        // Convert all observations to map coordinates
        vector<LandmarkObs> obs_landmarks;
        for (auto&& obs: observations) {
            // Convert observation from particle(vehicle) to map coordinate system
            LandmarkObs landmark;
            landmark.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta) ;
            landmark.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta) ;
            obs_landmarks.push_back(landmark);
        }
		
		
        // filter landmarks within sensor range
        vector<LandmarkObs> predicted_landmarks;
        for (const auto& landmark : map_landmarks.landmark_list) {

			// optimal way of comparing distance using lazy evaluation
            if ((p.x-landmark.x_f < sensor_range) &&
				(p.y-landmark.y_f < sensor_range) &&
				(sq(p.x-landmark.x_f) + sq(p.y-landmark.y_f) < sq(sensor_range))) {
				
                LandmarkObs l_pred;
                l_pred.id = landmark.id_i;
                l_pred.x = landmark.x_f;
                l_pred.y = landmark.y_f;
                predicted_landmarks.push_back(l_pred);
            }
        }
//		cout << "Particle (" << p.id << "," << p.x << "," << p.y << "," << p.theta << "," << p.weight << "), \t";
//		cout << "has " << predicted_landmarks.size() << " nearby landmarks" << endl;
		
        // Find which observations correspond to which landmarks (associate ids)
        dataAssociation(predicted_landmarks, obs_landmarks);

		
        // Compute the likelihood for each particle, that is the probablity of obtaining
        // current observations being in state (particle_x, particle_y, particle_theta)
        double particle_likelihood = 1.0;

        double mu_x, mu_y;
        for (const auto& obs : obs_landmarks) {

            // Find corresponding landmark on map for centering gaussian distribution
			//	this is suboptimal for dense landmarks; we could sort predicted landmarks by id and do binary search, or hash-table access
			LandmarkObs nearest;
            if (false) {  //optimized, but somehow giving error
				int it=obs.id; nearest.id = it; 
				nearest.x = map_landmarks.landmark_list[it].x_f; 
				nearest.y = map_landmarks.landmark_list[it].y_f; 
			}
			else 
			  for (const auto& land: predicted_landmarks)
                if (land.id == obs.id) {
					nearest = land;
                    break;
                }
//			cout << "nearest here = " << nearest.id << "," << nearest.x << "," << nearest.y << endl;

            //double prob = exp( -( sq(obs.x - mu_x) / (2 * sq(std_x)) + sq(obs.y - mu_y) / (2 * sq(std_y)) ) );
            //particle_likelihood *= prob / 2 * M_PI * std_x * std_y;
			double p1 = norm_prob((obs.x - nearest.x)/std_landmark[0])/std_landmark[0];
			double p2 = norm_prob((obs.y - nearest.y)/std_landmark[1])/std_landmark[1];
//			cout << "weight for obs=" << obs.x << "," << obs.y << " and " << nearest.x << "," << nearest.y << " is " << p1*p2 << endl;
			particle_likelihood *= p1 * p2;
        }
        p.weight = particle_likelihood; 
///			if (particle_likelihood!=0) cout << "weight=" << particle_likelihood << endl;
	}

	
	//normalize the weights
    double sum_wts = 0.0;
    for (const auto& particle : particles)
        sum_wts += particle.weight;
	if (sum_wts==0) sum_wts = numeric_limits<double>::epsilon();
    for (auto& particle : particles)
        particle.weight /= sum_wts;
	
	cout << "with updated Weights ... \t" << endl;
	logParticles(particles);
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	//create weights vector for calling discrete_distribution
	weights.clear();
    for (const auto& particle : particles)
        weights.push_back(particle.weight);

	// Take a discrete distribution with pmf equal to weights
    discrete_distribution<int> weighted_distribution(weights.begin(), weights.end());
    // initialise new particle array
    vector<Particle> newParticles;
    // resample particles
    for (int i = 0; i < num_particles; ++i) {
		int k = weighted_distribution(gen);
        newParticles.push_back(particles[k]);
	}
    particles = newParticles;
	cout << "Post Resampling, unique Particles left = " << countUnique(newParticles) << endl;
	logParticles(newParticles);
    for (int i = 0; i < num_particles; ++i) {
		newParticles[i].weight = 1.0;	// reset weights; these are again updated before next iteration
		newParticles[i].id = i;			// reset to unique ids, as associated predicted_particle against each observation is searched by id
	}
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}


template <typename T> string listVector(vector<T> v)
{
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getAssociations(Particle best) {
	return listVector(best.associations);
}
string ParticleFilter::getSenseX(Particle best) {
	return listVector(best.sense_x);
}
string ParticleFilter::getSenseY(Particle best) {
	return listVector(best.sense_y);
}
