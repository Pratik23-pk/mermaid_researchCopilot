from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse


class MermaidBrandingTests(TestCase):
    def test_landing_page_uses_mermaid_brand(self) -> None:
        response = self.client.get(reverse("landing"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Mermaid")

    def test_workspace_page_uses_mermaid_brand(self) -> None:
        user = User.objects.create_user(username="intern", password="secret123")
        self.client.force_login(user)

        response = self.client.get(reverse("workspace"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ask Mermaid")
