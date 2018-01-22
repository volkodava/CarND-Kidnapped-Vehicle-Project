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
#include "float.h"

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Number of particles to draw
  num_particles = 100;

  random_device rd;
  default_random_engine gen(rd());

  // Normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std[0]);
  // Normal distributions for y and theta
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Initialize particles from the Normal (Gaussian) distribution
  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
  }

  weights.resize(num_particles, 0.0);

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  random_device rd;
  default_random_engine gen(rd());

  // Normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(0.0f, std_pos[0]);
  // Normal distributions for y and theta
  normal_distribution<double> dist_y(0.0f, std_pos[1]);
  normal_distribution<double> dist_theta(0.0f, std_pos[2]);

  for (int i = 0; i < num_particles; ++i) {
    if (fabs(yaw_rate) < EPS) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
      // final yaw_rate == initial yaw_rate, because yaw_rate = 0
    } else {
      const double yaw_rate_per_t = yaw_rate * delta_t;
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate_per_t) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate_per_t));
      particles[i].theta += yaw_rate_per_t;
    }

    // Add random Gaussian noise to the particles
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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

  const double sig_x = std_landmark[0];
  const double sig_y = std_landmark[1];

  // calculate normalization term
  const double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
  const double sig_x_denom = 2 * pow(sig_x, 2);
  const double sig_y_denom = 2 * pow(sig_y, 2);

  for (int i = 0; i < num_particles; ++i) {
    double mgp = 1.0;

    for (int j = 0; j < observations.size(); ++j) {
      const double x_obs = particles[i].x + cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y;
      const double y_obs = particles[i].y + sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y;

      double min_dist = DBL_MAX;
      int min_index = -1;

      const std::vector<Map::single_landmark_s> landmark_list = map_landmarks.landmark_list;

      for (int k = 0; k < landmark_list.size(); ++k) {
        Map::single_landmark_s landmark = landmark_list[k];
        const double landmark_dist = dist(particles[i].x, particles[i].y, landmark.x_f, landmark.y_f);

        if (landmark_dist <= sensor_range) {
          const double cur_dist = dist(x_obs, y_obs, landmark.x_f, landmark.y_f);

          if (cur_dist < min_dist) {
            min_dist = cur_dist;
            min_index = k;
          }
        }
      }

      if (min_index < 0) {
        mgp = 0.0;
      } else {
        const float mu_x = landmark_list[min_index].x_f;
        const float mu_y = landmark_list[min_index].y_f;

        // calculate exponent
        const double exponent = pow(x_obs - mu_x, 2) / sig_x_denom + pow(y_obs - mu_y, 2) / sig_y_denom;

        // calculate weight using normalization terms and exponent
        mgp *= (gauss_norm * exp(-exponent));
      }
    }

    particles[i].weight = mgp;
    weights[i] = mgp;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::vector<Particle> resampled_particles;

  random_device rd;
  default_random_engine gen(rd());

  discrete_distribution<int> index(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; ++i) {
    resampled_particles.push_back(particles[index(gen)]);
  }

  particles = move(resampled_particles);
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
