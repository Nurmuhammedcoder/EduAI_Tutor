import requests

class CanvasAPI:
    def __init__(self, base_url, api_key):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    def test_connection(self):
        url = f"{self.base_url}/api/v1/users/self"
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return True, response.json().get('name', '')
            return False, f"Ошибка: {response.status_code} - {response.text}"
        except Exception as e:
            return False, str(e)
    
    def get_courses(self):
        url = f"{self.base_url}/api/v1/courses"
        params = {"enrollment_state": "active", "per_page": 50}
        try:
            courses = []
            while url:
                response = requests.get(url, headers=self.headers, params=params)
                if response.status_code == 200:
                    for course in response.json():
                        if 'name' in course:
                            courses.append({
                                'id': course['id'],
                                'name': course['name']
                            })
                    links = response.links
                    url = links['next']['url'] if 'next' in links else None
                else:
                    return False, f"Ошибка: {response.status_code} - {response.text}"
            return True, courses
        except Exception as e:
            return False, str(e)
    
    def get_assignments(self, course_id):
        url = f"{self.base_url}/api/v1/courses/{course_id}/assignments"
        try:
            assignments = []
            while url:
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    for assignment in response.json():
                        assignments.append({
                            'id': assignment['id'],
                            'name': assignment['name'],
                            'description': assignment.get('description', ''),
                            'due_at': assignment.get('due_at', '')
                        })
                    links = response.links
                    url = links['next']['url'] if 'next' in links else None
                else:
                    return False, f"Ошибка: {response.status_code} - {response.text}"
            return True, assignments
        except Exception as e:
            return False, str(e)
    
    def upload_grade(self, course_id, assignment_id, student_id, grade, comment):
        url = f"{self.base_url}/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions/{student_id}"
        data = {
            "comment": {"text_comment": comment},
            "submission": {"posted_grade": grade}
        }
        try:
            response = requests.put(url, headers=self.headers, json=data)
            if response.status_code == 200:
                return True, "Оценка успешно загружена"
            return False, f"Ошибка: {response.status_code} - {response.text}"
        except Exception as e:
            return False, str(e)