class Restaurant(object):

    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type

    def open_restaurant(self):
        print('Restaurant: %s, is opened.' % self.restaurant_name)

    def describe_restaurant(self):
        print('[Name: %s, Cuisine Type: %s]' % (self.restaurant_name, self.cuisine_type))


restaurant = Restaurant('asdjfhsld', 'Unknown')

print(restaurant.restaurant_name)
print(restaurant.cuisine_type)

restaurant.describe_restaurant()
restaurant.open_restaurant()
