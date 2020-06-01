curl -X POST --header 'Content-Type: application/json' --header 'Accept: application/json' -d @rl_reasoner_api_test_2.json 'http://localhost:9090/query'

curl -X POST --header 'Content-Type: application/json' --header 'Accept: application/json' -d @rl_reasoner_api_test_2.json 'https://indigo.ncats.io/rlr/api/v0/query'
