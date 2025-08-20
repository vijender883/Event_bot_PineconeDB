# How to Contribute: Forking and Pull Requests

Contributing to open-source projects like the RAG Event Bot is a great way to learn and collaborate. This guide will walk you through the process of forking the repository, making your own changes, and submitting them for review.

## 1. Fork the Repository

First, you need to create your own copy of the project on GitHub. This is called "forking."

1.  Navigate to the main repository page on GitHub.
2.  In the top-right corner of the page, click the **Fork** button.
3.  GitHub will create a copy of the repository under your account.

You now have a personal copy of the project!

## 2. Clone Your Fork

Next, you need to download your forked repository to your local machine. This is called "cloning."

1.  On your forked repository's GitHub page, click the green **Code** button.
2.  Copy the URL provided (it should look something like `https://github.com/YOUR_USERNAME/Event_bot_PineconeDB`).
3.  Open your terminal or command prompt and run the following command:

    ```bash
    git clone https://github.com/YOUR_USERNAME/Event_bot_PineconeDB
    ```

    Replace `YOUR_USERNAME` with your actual GitHub username.

4.  Navigate into the newly created project directory:
    ```bash
    cd RAG-Event-Bot
    ```

## 3. Create a New Branch

It's important to create a new branch for each new feature or bug fix you work on. This keeps your changes organized and separate from the main codebase.

Choose a descriptive name for your branch. For example, if you're adding a new feature, you could name your branch `feature/new-cool-feature`.

```bash
git checkout -b feature/your-new-feature-name
```

This command creates a new branch and switches to it.

## 4. Make Your Changes

Now you can start making your changes to the code. Use your favorite code editor to modify the files.

## 5. Commit Your Changes

Once you're happy with your changes, you need to "commit" them. A commit is like a snapshot of your changes at a specific point in time.

1.  **Stage your changes:**
    ```bash
    git add .
    ```
    This command adds all the files you've changed to the staging area.

2.  **Commit your changes:**
    ```bash
    git commit -m "feat: Add a clear and concise commit message"
    ```
    Write a short, descriptive message that explains the change you made.

## 6. Push Your Changes to GitHub

Now, you need to upload your changes to your forked repository on GitHub.

```bash
git push origin feature/your-new-feature-name
```

Replace `feature/your-new-feature-name` with the name of your branch.

## 7. Open a Pull Request

The final step is to create a "Pull Request" (PR). A pull request tells the original project maintainers that you have some changes you'd like to merge into the main project.

1.  Go to your forked repository on GitHub.
2.  You should see a banner with your recently pushed branch. Click the **"Compare & pull request"** button.
3.  You'll be taken to the "Open a pull request" page.
4.  Give your pull request a clear title and a detailed description of the changes you made.
5.  Click the **"Create pull request"** button.

That's it! You've successfully submitted a pull request. The project maintainers will review your changes and may ask for some modifications before merging them. Thank you for your contribution!
