#include <iostream>
#include <string>
#include <cstring>
using namespace std;

// define what item in hash table is

typedef struct Ht_item Ht_item;

struct Ht_item{
	char* key;
	char* value;
};

typedef struct HashTable HashTable;

struct HashTable{
	Ht_item** items;
	int count;
	int size;
};

unsigned long hash_function(char* str){
	unsigned long i = 0;
	for (int j=0; str[j]; j++){
		i+=str[j];
	}
	return i % 100;// capacity
}


//create block of memory to store hash table item
Ht_item* create_item(char* key, char* value){
	Ht_item* item = (Ht_item*) malloc(sizeof(Ht_item));
	item->key = (char*) malloc(strlen(key)+1);
	item->value = (char*) malloc(strlen(value)+1);
	
	strcpy(item->key, key);
	strcpy(item->value, value);
	
	return item;
}
