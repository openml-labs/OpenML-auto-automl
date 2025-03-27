# Setup for Openml Test Server
- The main reason we need this is because Snellius does not allow for cron jobs.
- If we eventually find a better solution that is not Snellius, we should eventually
migrate to a different platform. This is more of a temporary solution than a permanent
one.

## Steps to follow
- The first thing to do is authenticate using ssh to snellius from the test server.
- Turn on your VPN, ssh into the test server using your credentials ` ssh
user@openml-test.win.tue.nl`
- I would switch to bash at this point by just typing `bash`
- Now for the slightly complicated part, we shall authenticate using ssh the credentials
for Snellius here. Note that you should avoid using a password when it asks you to create
one. (YES THIS IS UNSAFE. We really need to find a better solution)
    - Check if the ssh agent is running `eval "$(ssh-agent -s)"`
    - Generate an SSH key to connect to the server. Give it no password `ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""`
    - Now copy the SSH key over to Snellius `ssh-copy-id -i ~/.ssh/id_rsa.pub user@snellius.surf.nl`
    - See if this works without a password now `ssh 'user@snellius.surf.nl'`
    - Add the SSH to your SSH manager `ssh-add ~/.ssh/id_rsa`
- Once that is done, we set up a cron job to run the script every n hours (here 5)
    - This assumes a few things
        - On Snellius, you have cloned this repository in `/home/user/`
        - You have an api key from OpenML
    - Type `crontab -e` and paste this line at the end
        - `0 */5 * * * ssh -i /home/user/.ssh/id_rsa user@snellius.surf.nl "bash -c
        '/home/user/OpenML-auto-automl/snellius_env_loader_cron.sh && python
    /home/user/OpenML-auto-automl/src/snellius_generate_sbatch.py -c -a
    apikey'" >> /home/user/cron_log.txt 2>&1`
    - The logs are stored in `/home/user/cron_log.txt`

