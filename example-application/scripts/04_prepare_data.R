## ------------------------------------------------------------------------
##
## Prepare data from Hake et al. (2024) for model fitting.
## 
## Author: Maarten van der Velde
## 
## Date: 2026-05-20
##
## ------------------------------------------------------------------------


library(here)
library(data.table)
library(janitor)
library(forcats)


# Load data ---------------------------------------------------------------

load(here("data", "raw", "hake2024.Rdata"))


# Format data -------------------------------------------------------------

d_full <- clean_names(bk) |>
  as.data.table()

d <- d_full[, .(user_id = fct_anon(user_id, prefix = "user_"), 
                clinical_status,
                session_id = fct_anon(session_id, prefix = "session_"),
                lesson_id = fct_anon(as.character(lesson_id), prefix = "lesson_"),
                fact_id = fct_anon(as.character(fact_id), prefix = "fact_"),
                repetition,
                start_time = presentation_start_time/1000,
                rt = reaction_time/1000,
                correct)]

# Add a chronological session counter (counting across users)
setorder(d, user_id, start_time)
d[, session := .GRP, by = .(user_id, session_id)]


# Save data ---------------------------------------------------------------

fwrite(d, here("data", "processed", "hake2024.csv"))