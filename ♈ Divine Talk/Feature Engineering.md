**Things missing in admin dashboard:**
- Per minute rate of astrologer
- Rating count of astrologers
	508 / 1079 unrated astrologers
	199 / 684 unrated active astrologers
	29 % unrated active astrologers
- Category(Tarot card, Vedic)
- ~~Speciality(Love, Marriage, Career, Health)
	`astro_specialitie_pivots`, `astrologer_specialities`~~
- Astrologer followers

**Features to consider while ranking:**
- Is busy (Will need to train on live data)
- What hour of the day?
	Average number of astrologers online at that particular hour of the day
	Orders accepted by an astrologer at that hour of the day
- What day of the week?
- Average call and chat length of astrologer

**Doubts:**
- ~~Table `notp_callback_response`~~
- Table `agora_call_logs`
- Confusion between `chat_logs, agora_call_logs, exotel_customer_call_logs, orders` ✅
- ~~Table `track_chat_notifications`~~
- What is `role_id?` ✅
- Repurchase Rate, Available for specific activities