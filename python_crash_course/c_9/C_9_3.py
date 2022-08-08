class User(object):

    def __init__(self, first_name, last_name, **data):
        self.first_name = first_name
        self.last_name = last_name
        self.data = data

    def describe_user(self):
        full_name = self.first_name.title() + ' ' + self.last_name.title()
        print('[user name: %s]' % full_name)

        for i in self.data:
            print('[key: %s, value: %s]' % (i, self.data[i]))

    def greet_user(self):
        print('Greetings, Mr.%s' % self.first_name.title())


user_1 = User(first_name='Bear', last_name='Grills', job='UK army', age='40', height='180cm')
user_1.greet_user()
