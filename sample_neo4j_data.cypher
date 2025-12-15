// Sample FPL Data for Neo4j
// Copy-paste these queries into Neo4j Browser to quickly populate your database

// ========================================
// CLEAR EXISTING DATA (OPTIONAL - BE CAREFUL!)
// ========================================
// MATCH (n) DETACH DELETE n

// ========================================
// CREATE PLAYERS - FORWARDS
// ========================================

CREATE (p1:Player {
  name: "Erling Haaland",
  position: "FWD",
  team: "Man City",
  total_points: 196,
  price: 14.0,
  goals: 27,
  assists: 5,
  clean_sheets: 0,
  minutes: 2160,
  bonus: 15,
  form: 8.5,
  selected: 5200000
})

CREATE (p2:Player {
  name: "Harry Kane",
  position: "FWD",
  team: "Tottenham",
  total_points: 178,
  price: 11.5,
  goals: 21,
  assists: 3,
  clean_sheets: 0,
  minutes: 2250,
  bonus: 12,
  form: 7.2,
  selected: 3100000
})

CREATE (p3:Player {
  name: "Ivan Toney",
  position: "FWD",
  team: "Brentford",
  total_points: 145,
  price: 8.5,
  goals: 15,
  assists: 4,
  clean_sheets: 0,
  minutes: 1980,
  bonus: 8,
  form: 6.8,
  selected: 1800000
})

CREATE (p4:Player {
  name: "Darwin Nunez",
  position: "FWD",
  team: "Liverpool",
  total_points: 132,
  price: 9.0,
  goals: 14,
  assists: 6,
  clean_sheets: 0,
  minutes: 1890,
  bonus: 6,
  form: 6.5,
  selected: 1500000
})

// ========================================
// CREATE PLAYERS - MIDFIELDERS
// ========================================

CREATE (p5:Player {
  name: "Mohamed Salah",
  position: "MID",
  team: "Liverpool",
  total_points: 188,
  price: 13.0,
  goals: 19,
  assists: 12,
  clean_sheets: 0,
  minutes: 2340,
  bonus: 20,
  form: 7.8,
  selected: 4200000
})

CREATE (p6:Player {
  name: "Kevin De Bruyne",
  position: "MID",
  team: "Man City",
  total_points: 165,
  price: 12.5,
  goals: 7,
  assists: 16,
  clean_sheets: 0,
  minutes: 2100,
  bonus: 18,
  form: 7.5,
  selected: 2900000
})

CREATE (p7:Player {
  name: "Heung-Min Son",
  position: "MID",
  team: "Tottenham",
  total_points: 154,
  price: 11.0,
  goals: 12,
  assists: 9,
  clean_sheets: 0,
  minutes: 2250,
  bonus: 14,
  form: 7.0,
  selected: 2400000
})

CREATE (p8:Player {
  name: "Phil Foden",
  position: "MID",
  team: "Man City",
  total_points: 142,
  price: 9.5,
  goals: 11,
  assists: 8,
  clean_sheets: 0,
  minutes: 1950,
  bonus: 12,
  form: 6.8,
  selected: 2100000
})

CREATE (p9:Player {
  name: "Bruno Fernandes",
  position: "MID",
  team: "Man Utd",
  total_points: 138,
  price: 10.0,
  goals: 8,
  assists: 10,
  clean_sheets: 0,
  minutes: 2400,
  bonus: 10,
  form: 6.5,
  selected: 2000000
})

// ========================================
// CREATE PLAYERS - DEFENDERS
// ========================================

CREATE (p10:Player {
  name: "Trent Alexander-Arnold",
  position: "DEF",
  team: "Liverpool",
  total_points: 152,
  price: 7.5,
  goals: 2,
  assists: 11,
  clean_sheets: 12,
  minutes: 2340,
  bonus: 16,
  form: 7.2,
  selected: 3500000
})

CREATE (p11:Player {
  name: "Reece James",
  position: "DEF",
  team: "Chelsea",
  total_points: 135,
  price: 6.5,
  goals: 3,
  assists: 8,
  clean_sheets: 10,
  minutes: 2100,
  bonus: 12,
  form: 6.8,
  selected: 2800000
})

CREATE (p12:Player {
  name: "Kieran Trippier",
  position: "DEF",
  team: "Newcastle",
  total_points: 142,
  price: 6.0,
  goals: 1,
  assists: 9,
  clean_sheets: 15,
  minutes: 2250,
  bonus: 14,
  form: 7.0,
  selected: 3200000
})

CREATE (p13:Player {
  name: "Ben White",
  position: "DEF",
  team: "Arsenal",
  total_points: 128,
  price: 5.5,
  goals: 1,
  assists: 6,
  clean_sheets: 14,
  minutes: 2400,
  bonus: 10,
  form: 6.5,
  selected: 2500000
})

// ========================================
// CREATE PLAYERS - GOALKEEPERS
// ========================================

CREATE (p14:Player {
  name: "Nick Pope",
  position: "GKP",
  team: "Newcastle",
  total_points: 145,
  price: 5.5,
  goals: 0,
  assists: 0,
  clean_sheets: 18,
  minutes: 2520,
  bonus: 8,
  form: 7.5,
  selected: 2800000
})

CREATE (p15:Player {
  name: "Aaron Ramsdale",
  position: "GKP",
  team: "Arsenal",
  total_points: 138,
  price: 5.0,
  goals: 0,
  assists: 0,
  clean_sheets: 16,
  minutes: 2520,
  bonus: 6,
  form: 7.0,
  selected: 2400000
})

CREATE (p16:Player {
  name: "Alisson Becker",
  position: "GKP",
  team: "Liverpool",
  total_points: 132,
  price: 5.5,
  goals: 0,
  assists: 1,
  clean_sheets: 14,
  minutes: 2430,
  bonus: 7,
  form: 6.8,
  selected: 2200000
})

// ========================================
// VERIFY DATA
// ========================================

// Count players by position
MATCH (p:Player)
RETURN p.position as position, count(p) as count
ORDER BY position

// Show all players
MATCH (p:Player)
RETURN p.name, p.position, p.team, p.total_points, p.price
ORDER BY p.total_points DESC

// Check specific player
MATCH (p:Player {name: "Erling Haaland"})
RETURN p

// ========================================
// OPTIONAL: CREATE TEAMS (for relationships)
// ========================================

CREATE (t1:Team {name: "Man City", short_name: "MCI"})
CREATE (t2:Team {name: "Liverpool", short_name: "LIV"})
CREATE (t3:Team {name: "Arsenal", short_name: "ARS"})
CREATE (t4:Team {name: "Tottenham", short_name: "TOT"})
CREATE (t5:Team {name: "Newcastle", short_name: "NEW"})
CREATE (t6:Team {name: "Man Utd", short_name: "MUN"})
CREATE (t7:Team {name: "Chelsea", short_name: "CHE"})
CREATE (t8:Team {name: "Brentford", short_name: "BRE"})

// ========================================
// OPTIONAL: CREATE RELATIONSHIPS
// ========================================

MATCH (p:Player), (t:Team)
WHERE p.team = t.name
CREATE (p)-[:PLAYS_FOR]->(t)

// Verify relationships
MATCH (p:Player)-[r:PLAYS_FOR]->(t:Team)
RETURN p.name, type(r), t.name
LIMIT 10

// ========================================
// QUERIES TO TEST YOUR SYSTEM
// ========================================

// Test Query 1: Top forwards
MATCH (p:Player)
WHERE p.position = 'FWD'
RETURN p.name, p.team, p.total_points, p.price
ORDER BY p.total_points DESC
LIMIT 5

// Test Query 2: Players under 8 million
MATCH (p:Player)
WHERE p.price <= 8.0 AND p.position = 'MID'
RETURN p.name, p.team, p.price, p.total_points
ORDER BY p.total_points DESC

// Test Query 3: Players by team
MATCH (p:Player)
WHERE p.team = 'Man City'
RETURN p.name, p.position, p.total_points
ORDER BY p.total_points DESC

// Test Query 4: Clean sheet leaders
MATCH (p:Player)
WHERE p.position IN ['GKP', 'DEF']
RETURN p.name, p.team, p.clean_sheets
ORDER BY p.clean_sheets DESC
LIMIT 5

// ========================================
// NOTES
// ========================================

/*
This creates 16 sample players:
- 4 Forwards
- 5 Midfielders
- 4 Defenders
- 3 Goalkeepers

This is the MINIMUM data you need for testing.

For a complete system, you'd want:
- 50+ players
- All Premier League teams
- Historical gameweek data
- More detailed statistics

But this sample data is enough to:
✓ Test all Cypher queries
✓ Demo the system
✓ Pass the evaluation

*/
