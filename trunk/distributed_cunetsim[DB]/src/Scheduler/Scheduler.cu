#ifndef STRUCTURES_H_
#define STRUCTURES_H_
#include "../structures.h"
#include "/usr/local/cuda/include/curand.h"
#endif /* STRUCTURES_H_ */
#ifndef INTERFACES_H_
#define INTERFACES_H_
#include "../interfaces.h"
#endif /* INTERFACES_H_ */
#include "../vars.h"

#include <deque>
#define MAXLPS 50 

using namespace std;

// This list is going to contain the simulation events
deque<struct Event> event_list;

void Schedule (int first_round, int order_in_round, int nb_rounds, enum Event_Distribution distribution, int frequency_in_round, 
enum Event_Type event_type, int nb_tours)
{
	struct Event new_event;
	new_event.type = event_type;
	new_event.frequency = frequency_in_round;
	new_event.order = order_in_round;
	
	if (distribution == CONSEC){
		for (int round = first_round; (round < first_round + nb_rounds && round < nb_tours) ; round ++){
			new_event.round = round;
			
			// now that the event data is ready, we look for the right position in the list where we must put it
			deque<struct Event>::iterator it;
			it = event_list.begin();
			while (it->round < round && it != event_list.end())
				it++;
			if (it->round > round)
				event_list.insert(it, new_event);
			else {
				while (it->order < order_in_round && it->round == round && it != event_list.end())
					it++;
				event_list.insert(it, new_event);
			}
		}
	} else if (distribution == DISTRIB){
		for (int round = first_round; round < (nb_tours - (nb_tours % nb_rounds)); round += nb_tours/nb_rounds){
			new_event.round = round;
			
			// now that the event data is ready, we look for the right position in the list where we must put it
			deque<struct Event>::iterator it;
			it = event_list.begin();
			while (it->round < round && it != event_list.end())
				it++;
			if (it->round > round)
				event_list.insert(it, new_event);
			else {
				while (it->order < order_in_round && it->round == round && it != event_list.end())
					it++;
				event_list.insert(it, new_event);
			}
		}
	}
}

void Calculate_timestamps (Simulation_Parameters simulation_parameters) {
	int nb_rounds = simulation_parameters.simulation_config.simulation_time + 100;
	deque<struct Event>::iterator it;
	
	int id = 0;
	it = event_list.begin();
	while (it != event_list.end()){
		it->timestamp = id;
		id++;
		it++;
	}
}

void Print_event_list () {

	for(int i(0); i<event_list.size(); ++i)
        printf("Event: %d, round %d, order %d, freq %d, ts %d\n", event_list[i].type, event_list[i].round, event_list[i].order, 
                  event_list[i].frequency, event_list[i].timestamp);
}

bool Next_event_of_the_round (struct Event * event, int round_num) {
	deque<struct Event>::iterator it;
	it = event_list.begin();
	
	if (it->round == round_num){
		event->type = it->type;
		event->frequency = it->frequency;
		event->timestamp = it->timestamp;
		event_list.pop_front();
		return true;
	}
	return false;
}
