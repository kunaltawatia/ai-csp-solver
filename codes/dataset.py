import random

levels = []
course_idx = 1
for level in range(8):
    level_courses = list(range(course_idx, course_idx + 30))
    levels.append(level_courses)
    course_idx += 30
total_courses = sum(len(level_courses) for level_courses in levels)

matrix = [
    [0 for j in range(total_courses + 1)]
    for i in range(total_courses + 1)
]

ug_students = 1000
pg_students = 600
faculty = 100


def add_choice(choice: list):
    for course_u in choice:
        for course_v in choice:
            matrix[course_u][course_v] = 1


allowed_courses = []

# first year
allowed_courses = levels[0] + levels[1]
for student in range(ug_students // 4):
    choice = random.choices(allowed_courses, k=2)
    add_choice(choice)

# second year
allowed_courses += levels[2]
for student in range(ug_students // 4):
    choice = random.choices(allowed_courses, k=2)
    add_choice(choice)

# third year
allowed_courses += levels[3] + levels[5]
for student in range(ug_students // 4):
    choice = random.choices(allowed_courses, k=3)
    add_choice(choice)

# fourth year
allowed_courses += levels[6]
for student in range(ug_students // 4):
    choice = random.choices(allowed_courses, k=3)
    add_choice(choice)

# pg msc
allowed_courses = levels[4] + levels[5] + levels[6]
for student in range(pg_students // 3):
    choice = random.choices(allowed_courses, k=2)
    add_choice(choice)

# pg mtech
allowed_courses += levels[7]
for student in range(pg_students // 3):
    choice = random.choices(allowed_courses, k=2)
    add_choice(choice)

# pg phd
allowed_courses = levels[7] + levels[6]
for student in range(pg_students // 3):
    choice = random.choices(allowed_courses, k=2)
    add_choice(choice)


def edges_from_matrix():
    edges = []

    for i in range(1, total_courses + 1):
        for j in range(i + 1, total_courses + 1):
            if(matrix[i][j]):
                edges.append([i, j])

    return edges

edges = edges_from_matrix()
print(len(edges))
for edge in edges_from_matrix():
    u, v = edge
    print(u, v)
